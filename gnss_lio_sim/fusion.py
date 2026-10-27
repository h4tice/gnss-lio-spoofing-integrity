"""Gevşek bağlı GNSS / LIO / IMU füzyonu (ESKF + stokastik klonlama) ve bütünlük denetimi.

* IMU ile 200 Hz durum yayılımı
* LIO göreli poz kısıtları (10 Hz): z = (R_{k-1}^T (p_k - p_{k-1}), R_{k-1}^T R_k)
  — önceki poz klonlanarak (21 boyutlu hata durumu) çapraz kovaryans korunur
* GNSS konum + hız güncellemeleri (5 Hz), kovaryansı geçiş politikasına göre şişirilir
* Tespit istatistikleri her GNSS çağında nominal (şişirilmemiş) kovaryansla hesaplanır
"""
import time
from collections import deque

import numpy as np
from scipy.stats import chi2

from .detection import IntegrityMonitor
from .geometry import quat_to_rot, rot_to_quat, skew, so3_log
from .ins import Ins, N_CORE
from .lio import degeneracy_flags, relative_cov

METHODS = ["gnss_ins", "monitor", "hard", "voting", "soft"]


def gnss_R(cfg):
    g = cfg.gnss
    var_h = g.sigma_h ** 2 + g.gm_sigma ** 2
    var_v = g.sigma_v ** 2 + g.get("gm_sigma_v", 1.5 * g.gm_sigma) ** 2
    Rp = np.diag([var_h, var_h, var_v]) * cfg.fusion.gnss_r_scale
    sv = g.get("vel_sigma_v", g.vel_sigma)
    Rv = np.diag([g.vel_sigma ** 2, g.vel_sigma ** 2, sv ** 2]) * cfg.fusion.gnss_r_scale
    return Rp, Rv


def run_fusion(flight, att, method, cfg, detect_only=False):
    """Bir uçuş kaydı + saldırı senaryosu üzerinde füzyonu çalıştırır.

    method: gnss_ins | monitor | hard | voting | soft | ci_ins | gu_lio | alja_lio (son üçü literatür
    karşılaştırmaları, bkz. baselines.py)
    detect_only: tespit istatistikleri hesaplanır ama GNSS hiç dışlanmaz / ağırlığı değiştirilmez
    (yöntemleri aynı süzgeç durumu üzerinde karşılaştıran ROC analizi için, bkz. roc.py)
    """
    from .baselines import AljaafrehMonitor, CIMonitor, GuMonitor
    use_lio = method not in ("gnss_ins", "ci_ins")
    policy = {"gnss_ins": "none"}.get(method, method)
    use_vel = cfg.fusion.use_gnss_velocity
    Rp, Rv = gnss_R(cfg)
    if method == "ci_ins":
        monitor = CIMonitor(cfg)
    elif method == "gu_lio":
        monitor = GuMonitor(cfg, Rp)
    elif method == "alja_lio":
        monitor = AljaafrehMonitor(cfg)
    else:
        monitor = IntegrityMonitor(cfg.detector, cfg.transition, cfg.gnss, policy,
                                   use_velocity=use_vel, use_disp=use_lio)

    ti = flight["imu_t"]
    acc, gyro = flight["imu_acc"], flight["imu_gyro"]
    R0 = quat_to_rot(flight["truth_q"][0])
    gi = 0
    # başlangıç: ilk GNSS konumu, bilinen başlangıç yönelimi (yer hizalaması), askıda
    P0 = Ins.default_P0(sig_p=np.sqrt(Rp[0, 0]), sig_v=0.1, sig_th_deg=1.0)
    ins = Ins(att["pos"][0], np.zeros(3), R0, P0, cfg.imu)

    lio_t = flight["lio_t"]
    lio_p = flight["lio_p"]
    lio_R = quat_to_rot(flight["lio_q"])
    lio_info = flight["lio_info6"]
    lio_deg = degeneracy_flags(flight["lio_pos_info_ev"][:, 0], cfg.lio, flight["lio_rot_info_ev"][:, 0])
    gt = att["t"]
    n_imu = len(ti)
    scan_every = int(round(cfg.imu.rate / cfg.lidar.rate))
    gnss_every = int(round(cfg.imu.rate / cfg.gnss.rate))

    out_t, out_p, out_v, out_q, out_P = [], [], [], [], []
    g_rec = {k: [] for k in ("t", "rA", "rB", "rC", "rD", "rho", "lam", "alarm", "nis_p", "nis_v")}
    # LIO birikimli yer değiştirmesi (dünya), kovaryansı ve dejenere tarama sayacı (D testi için)
    disp = {"D": np.zeros(3), "C": np.zeros((3, 3)), "n": 0, "ndeg": 0}
    disp["p"] = None  # ham LIO mutlak konumu (Gu ve ark. karşılaştırması; kovaryansı disp["C"])
    sig_deg2 = cfg.detector.disp_deg_sigma ** 2
    li = 0
    lio_gate = chi2.ppf(1 - cfg.fusion.lio_gate_alpha, 6)
    n_reject = n_lio = n_unhealthy = 0
    lio_win = deque(maxlen=cfg.fusion.lio_health_window)
    lio_health_thr = chi2.ppf(1 - cfg.fusion.lio_health_alpha, 3 * cfg.fusion.lio_health_window)
    lio_bad_until = lio_bad_from = -np.inf
    gnss_aid = {"t": -np.inf}  # GNSS'in en son fiilen kullanıldığı an (λ küçük)
    scan_dt = 1.0 / cfg.lidar.rate
    tic = time.perf_counter()
    upd_times = []
    lio_codes = []
    for k in range(1, n_imu):
        ins.propagate(acc[k - 1], gyro[k - 1], ti[k] - ti[k - 1])
        tk = ti[k]
        is_scan = k % scan_every == 0
        if is_scan and use_lio:
            # lio_t[li] == tk (ortak zaman tabanı)
            while li < len(lio_t) and lio_t[li] < tk - 1e-6:
                li += 1
            if li < len(lio_t) and abs(lio_t[li] - tk) < 1e-6:
                u0 = time.perf_counter()
                if ins.clone_p is not None and li > 0:
                    Rw = ins.clone_R
                    pred_disp = ins.p - ins.clone_p
                    args = (lio_p[li - 1], lio_R[li - 1], lio_p[li], lio_R[li])
                    # aykırı değer kapısı: IMU öngörüsüyle tutarsız LIO artımı önce yatay/yaw
                    # yönleri atılarak yeniden denenir, yine tutarsızsa reddedilir
                    # GNSS kesik/dışlanmışken LIO'nun tamamen dışlanması geriye yalnız IMU bırakır:
                    # bu durumda dışlama en fazla lio_unhealthy_hold_nognss saniye sürer
                    gnss_on = tk - gnss_aid["t"] < cfg.fusion.gnss_aid_window
                    gnss_off_since = gnss_aid["t"] + cfg.fusion.gnss_aid_window
                    unhealthy = tk < lio_bad_until and (
                        gnss_on or tk < max(gnss_off_since, lio_bad_from) + cfg.fusion.lio_unhealthy_hold_nognss)
                    deg = bool(lio_deg[li]) or unhealthy
                    if not lio_deg[li]:
                        # LIO sağlık denetimi: dejenere olmayan taramaların konum yeniliği NIS'i
                        # hız toleransı: füzyon hızındaki küçük (ör. aldatılmış GNSS hızından gelen)
                        # yanlılık LIO'yu sağlıksız gösterip aldatmayı maskelemesin
                        cov_full = relative_cov(lio_info[li], lio_R[li - 1], cfg.lio, False)
                        cov_full[:3, :3] += np.eye(3) * (cfg.fusion.lio_health_vel_tol * scan_dt) ** 2
                        lio_win.append(_lio_nis_pos(ins, *args, cov_full))
                        if len(lio_win) == lio_win.maxlen and sum(lio_win) > lio_health_thr and (
                                gnss_on or cfg.fusion.lio_unhealthy_hold_nognss > 0):
                            lio_bad_until = tk + cfg.fusion.lio_unhealthy_hold
                            lio_bad_from = tk
                            lio_win.clear()
                            unhealthy = deg = True
                    cov = relative_cov(lio_info[li], lio_R[li - 1], cfg.lio, deg)
                    if unhealthy:
                        ok = first_ok = False  # sağlıksız LIO: tamamen IMU-öncelikli mod
                    else:
                        ok = first_ok = _lio_update(ins, *args, cov, lio_gate)
                    if not ok and not deg and not unhealthy:
                        # LIO ıraksaması şüphesi: IMU-öncelikli moda geç (yalnız z/roll/pitch)
                        cov = relative_cov(lio_info[li], lio_R[li - 1], cfg.lio, True)
                        ok = _lio_update(ins, *args, cov, lio_gate)
                    n_unhealthy += int(unhealthy)
                    # tarama başına karar: 0 tam kabul, 1 dejenere kipte kabul, 2 red, 3 sağlıksız
                    lio_codes.append((tk, 3 if unhealthy else (0 if first_ok and not deg else (1 if ok else 2))))
                    disp["n"] += 1
                    disp["p"] = lio_p[li]
                    if first_ok and not unhealthy:
                        # D testi GNSS'ten bağımsız olmalı: LIO artımı LIO'nun kendi çerçevesinde
                        # (başlangıçta ENU'ya hizalı) birikir; füzyon yönelimi kullanılmaz, çünkü
                        # aldatılmış GNSS hızı füzyonun yaw'ını bozup aldatmayı kısmen telafi edebilir
                        Rl0 = lio_R[li - 1]
                        disp["D"] = disp["D"] + (lio_p[li] - lio_p[li - 1])
                        Cw = Rl0 @ cov[:3, :3] @ Rl0.T
                        if deg:
                            disp["ndeg"] += 1
                        # tek bir zayıf taramanın önseli testi uzun süre kör etmesin: katkı sınırlı
                        ev, V = np.linalg.eigh(Cw)
                        disp["C"] = disp["C"] + (V * np.minimum(ev, sig_deg2)) @ V.T
                    else:
                        # LIO yatay artımı güvenilmez: D testinde dejenere say
                        n_reject += 1
                        disp["ndeg"] += 1
                        disp["D"] = disp["D"] + pred_disp
                        disp["C"] = disp["C"] + np.eye(3) * sig_deg2
                    n_lio += 1
                ins.set_clone()
                upd_times.append(time.perf_counter() - u0)
        if k % gnss_every == 0:
            while gi < len(gt) and gt[gi] < tk - 1e-6:
                gi += 1
            if gi < len(gt) and abs(gt[gi] - tk) < 1e-6:
                lam = _gnss_step(ins, att, gi, tk, monitor, Rp, Rv, use_vel, g_rec, disp if use_lio else None,
                                 detect_only)
                if lam is not None and lam < cfg.fusion.gnss_aid_lam:
                    gnss_aid["t"] = tk
        if is_scan:
            out_t.append(tk)
            out_p.append(ins.p.copy())
            out_v.append(ins.v.copy())
            out_q.append(ins.R.copy())
            out_P.append(np.diag(ins.P)[0:3].copy())
    wall = time.perf_counter() - tic
    res = {
        "t": np.array(out_t), "p": np.array(out_p), "v": np.array(out_v),
        "q": rot_to_quat(np.array(out_q)), "Pp": np.array(out_P),
        "wall": wall, "lio_update_ms": 1000 * np.mean(upd_times) if upd_times else 0.0,
        "lio_reject_rate": n_reject / max(n_lio, 1),
        "lio_unhealthy_rate": n_unhealthy / max(n_lio, 1),
        "lio_codes": np.array(lio_codes, dtype=float).reshape(-1, 2),
    }
    res.update({f"g_{k}": np.array(v, dtype=float) for k, v in g_rec.items()})
    return res


def _lio_nis_pos(ins, p0, R0, p1, R1, cov):
    """LIO göreli konum artımının füzyon öngörüsüne göre NIS'i (3 sd; güncelleme yapmaz)."""
    z_p = R0.T @ (p1 - p0)
    Rc = ins.clone_R
    r = z_p - Rc.T @ (ins.p - ins.clone_p)
    H = np.zeros((3, ins.P.shape[0]))
    H[:, 0:3] = Rc.T
    H[:, 15:18] = -Rc.T
    H[:, 18:21] = skew(Rc.T @ (ins.p - ins.clone_p))
    S = H @ ins.P @ H.T + cov[:3, :3]
    return float(r @ np.linalg.solve(S, r))


def _lio_update(ins, p0, R0, p1, R1, cov, gate=np.inf):
    """LIO göreli poz güncellemesi; yenilik NIS'i kapıyı aşarsa uygulanmaz (False döner)."""
    z_p = R0.T @ (p1 - p0)
    z_R = R0.T @ R1
    Rc = ins.clone_R
    zh_p = Rc.T @ (ins.p - ins.clone_p)
    zh_R = Rc.T @ ins.R
    r = np.r_[z_p - zh_p, so3_log(zh_R.T @ z_R)]
    H = np.zeros((6, N_CORE + 6))
    H[0:3, 0:3] = Rc.T
    H[0:3, 15:18] = -Rc.T
    H[0:3, 18:21] = skew(zh_p)
    H[3:6, 6:9] = np.eye(3)
    H[3:6, 18:21] = -zh_R.T
    S = H @ ins.P @ H.T + cov
    if r @ np.linalg.solve(S, r) > gate:
        return False
    ins.update(H, r, cov)
    return True


def _gnss_step(ins, att, gi, tk, monitor, Rp, Rv, use_vel, rec, disp, detect_only=False):
    n = ins.P.shape[0]
    adv = att.get("adaptive")  # uyarlanabilir aldatıcı: ölçümü dedektörün son durumuna göre üretir
    if adv is not None:
        adv.apply(gi, tk)
    valid = att["valid"][gi]
    nis_p = nis_v = None
    if valid:
        k = 2 if monitor.horiz else 3  # yatay bütünlük kipinde testler yalnız yatay yenilikle
        Hp = np.zeros((3, n))
        Hp[:, 0:3] = np.eye(3)
        nu_p = att["pos"][gi] - ins.p
        S_p = ins.P[0:3, 0:3] + Rp
        nis_p = float(nu_p[:k] @ np.linalg.solve(S_p[:k, :k], nu_p[:k]))
        if use_vel:
            Hv = np.zeros((3, n))
            Hv[:, 3:6] = np.eye(3)
            nu_v = att["vel"][gi] - ins.v
            S_v = ins.P[3:6, 3:6] + Rv
            nis_v = float(nu_v[:k] @ np.linalg.solve(S_v[:k, :k], nu_v[:k]))
    if disp is not None and valid:
        lam, info = monitor.step(tk, nis_p, nis_v, att["pos"][gi], disp["D"], disp["C"],
                                 (disp["n"], disp["ndeg"]), p_lio=disp["p"], C_abs=disp["C"])
    else:
        lam, info = monitor.step(tk, nis_p, nis_v)
    if adv is not None:
        adv.observe(info)
    if detect_only:
        lam = 1.0
    if valid and np.isfinite(lam) and lam < 1e5:
        if use_vel:
            H = np.vstack([Hp, Hv])
            r = np.r_[nu_p, nu_v]
            Rm = np.zeros((6, 6))
            Rm[:3, :3] = Rp * lam
            Rm[3:, 3:] = Rv * lam
        else:
            H, r, Rm = Hp, nu_p, Rp * lam
        ins.update(H, r, Rm)
    rec["t"].append(tk)
    for key in ("rA", "rB", "rC", "rD", "rho"):
        rec[key].append(info[key])
    rec["lam"].append(min(lam, 1e9))
    rec["alarm"].append(float(info["alarm"]))
    rec["nis_p"].append(np.nan if nis_p is None else nis_p)
    rec["nis_v"].append(np.nan if nis_v is None else nis_v)
    return lam if valid else None
