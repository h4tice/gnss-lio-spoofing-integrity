"""Değerlendirme metrikleri: ATE/RPE, tespit gecikmesi, yanlış alarm, geçiş sıçraması,
GNSS'siz sürüklenme ve sürüklenme-geometri bölütleri."""
import numpy as np

from .geometry import quat_to_rot

WARMUP = 5.0  # [s] filtre oturma süresi; yanlış alarm hesabından dışlanır


def umeyama(src, dst, with_scale=False):
    """src -> dst için en küçük kareler benzerlik/katı dönüşüm (R, t, s)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    C = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate(est, ref, align=False):
    if align:
        R, t, s = umeyama(est, ref)
        est = (s * (R @ est.T)).T + t
    e = np.linalg.norm(est - ref, axis=1)
    return float(np.sqrt(np.mean(e ** 2)))


def rpe(t, est, ref, delta=1.0):
    """Zaman tabanlı göreli konum hatası (öteleme RMSE, [m])."""
    dt = np.median(np.diff(t))
    k = max(1, int(round(delta / dt)))
    de = (est[k:] - est[:-k]) - (ref[k:] - ref[:-k])
    return float(np.sqrt(np.mean(np.sum(de ** 2, axis=1))))


def reference_at(flight, t):
    idx = np.clip(np.searchsorted(flight["rtk_t"], t - 1e-6), 0, len(flight["rtk_t"]) - 1)
    return flight["rtk_p"][idx]


def _plane(cfg, x):
    """evaluation.horizontal açıksa ölçütler yatay düzlemde hesaplanır (düşey bileşen sıfırlanır)."""
    if cfg.evaluation.get("horizontal", False):
        x = x.copy()
        x[:, 2] = 0.0
    return x


def path_length(p):
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def eval_run(flight, att, res, cfg):
    meta = flight["meta"]
    t_on, t_off = att["window"]
    attacked = np.isfinite(t_on)
    # gerçek veride değerlendirme görev bölümüyle sınırlanır (meta["mission_end"]); tüm uçuş
    # üzerindeki yanlış alarm oranı ayrıca "far_full" olarak verilir
    warm = meta.get("warmup", WARMUP)
    end = meta.get("mission_end", np.inf)
    far_full = np.nan
    if np.isfinite(end):
        g_all, a_all = res["g_t"], res["g_alarm"] > 0
        h_all = (g_all > warm) & ((g_all < t_on) if attacked else True)
        far_full = float(a_all[h_all].mean()) if h_all.any() else np.nan
        keep, gk = res["t"] < end, res["g_t"] < end
        res = {**res, "t": res["t"][keep], "p": res["p"][keep],
               **{k: v[gk] for k, v in res.items() if k.startswith("g_")}}
    t = res["t"]
    ref = _plane(cfg, reference_at(flight, t))
    P = _plane(cfg, res["p"])
    err_vec = P - ref
    err = np.linalg.norm(err_vec, axis=1)
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(err.max()),
        "end_err": float(err[-1]),
        "rpe_1s": rpe(t, P, ref, cfg.evaluation.rpe_delta),
        "wall_time": res["wall"],
        "lio_update_ms": res["lio_update_ms"],
        "far_full": far_full,
    }
    gt, alarm, lam = res["g_t"], res["g_alarm"] > 0, res["g_lam"]
    # gerçek veride sistem, LIO-ENU hizalaması bittikten sonra işler durumdadır (meta["warmup"])
    healthy = (gt > warm) & ((gt < t_on) if attacked else True)
    out["fa_epochs"] = int(alarm[healthy].sum())
    out["healthy_epochs"] = int(healthy.sum())
    out["far"] = float(alarm[healthy].mean()) if healthy.any() else np.nan
    out["any_false_alarm"] = bool(alarm[healthy].any())
    # sağlıklı bölümdeki (taban) adım süreksizliği
    step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0) - np.diff(ref, axis=0), axis=1)]
    base = (t > warm) & ((t < t_on) if attacked else True)
    out["jump_base"] = float(step[base].max()) if base.any() else np.nan
    if not attacked:
        return out

    # ---- tespit
    win = (gt >= t_on) & (gt < t_off)
    hits = np.where(win & alarm)[0]
    out["detected"] = bool(len(hits))
    if len(hits):
        i = hits[0]
        out["t_alarm"] = float(gt[i])
        out["delay"] = float(gt[i] - t_on)
        gi = np.argmin(np.abs(att["t"] - gt[i]))
        out["offset_at_det"] = float(np.linalg.norm(att["offset"][gi]))
        out["err_at_det"] = float(err[np.argmin(np.abs(t - gt[i]))])
    else:
        out.update(t_alarm=np.nan, delay=np.nan, offset_at_det=np.nan, err_at_det=np.nan)
    deweight = np.where(win & (lam > 2.0))[0]
    out["deweight_delay"] = float(gt[deweight[0]] - t_on) if len(deweight) else np.nan

    # ---- geçiş: ek hata sıçraması ve adım süreksizliği
    pre = (t >= t_on - 10.0) & (t < t_on)
    pre_err = float(err[pre].mean()) if pre.any() else 0.0
    post = t >= t_on
    attack_w = (t >= t_on) & (t < t_off)
    rec_w = t >= t_off
    out["pre_err"] = pre_err
    out["peak_add_err"] = float(err[post].max() - pre_err)
    out["peak_add_err_attack"] = float(err[attack_w].max() - pre_err)
    out["peak_add_err_recovery"] = float(err[rec_w].max() - pre_err) if rec_w.any() else np.nan
    out["jump_post"] = float(step[post].max())
    out["jump_add"] = float(step[post].max() - out["jump_base"])
    out["rmse_post"] = float(np.sqrt(np.mean(err[post] ** 2)))

    # ---- GNSS'siz seyirde sürüklenme (kat edilen mesafeye oranla)
    i0 = np.argmin(np.abs(t - t_on))
    i1 = np.argmin(np.abs(t - t_off))
    dist = path_length(ref[i0:i1 + 1])
    out["denied_dist"] = dist
    out["drift_abs"] = float(np.linalg.norm(err_vec[i1] - err_vec[i0]))
    out["drift_pct"] = 100.0 * out["drift_abs"] / max(dist, 1e-6)
    out["err_end_denied"] = float(err[i1])
    return out


def eval_lio_only(flight, cfg):
    """GNSS füzyonsuz taban hat LIO: ATE (ham ve hizalı), RPE, 60 s pencere sürüklenmesi."""
    t = flight["lio_t"]
    p = _plane(cfg, flight["lio_p"])
    ref = _plane(cfg, reference_at(flight, t))
    err_vec = p - ref
    out = {
        "rmse": ate(p, ref),
        "ate_aligned": ate(p, ref, align=True),
        "rpe_1s": rpe(t, p, ref, cfg.evaluation.rpe_delta),
        "max_err": float(np.linalg.norm(err_vec, axis=1).max()),
        "end_err": float(np.linalg.norm(err_vec[-1])),
        "lio_proc_ms_mean": float(1000 * flight["lio_proc_time"].mean()),
        "lio_proc_ms_p95": float(1000 * np.percentile(flight["lio_proc_time"], 95)),
        "lio_proc_ms_max": float(1000 * flight["lio_proc_time"].max()),
    }
    # aldatma penceresiyle aynı aralıkta yalnız-LIO sürüklenmesi
    meta = flight["meta"]
    i0 = np.argmin(np.abs(t - meta["onset"]))
    i1 = np.argmin(np.abs(t - meta["onset"] - meta["denied"]))
    dist = path_length(ref[i0:i1 + 1])
    out["denied_dist"] = dist
    out["drift_abs"] = float(np.linalg.norm(err_vec[i1] - err_vec[i0]))
    out["drift_pct"] = 100.0 * out["drift_abs"] / max(dist, 1e-6)
    return out


def drift_segments(flight, cfg):
    """AS3: yalnız-LIO bölütlerinde birim mesafe başına sürüklenme + ortam geometrisi öznitelikleri."""
    t = flight["lio_t"]
    p = flight["lio_p"]
    R = quat_to_rot(flight["lio_q"])
    ref = reference_at(flight, t)
    seg = cfg.evaluation.drift_segment
    dt = np.median(np.diff(t))
    n = int(round(seg / dt))
    rows = []
    for i0 in range(int(WARMUP / dt), len(t) - n, n):
        i1 = i0 + n
        dist = path_length(ref[i0:i1 + 1])
        if dist < 5.0:
            continue
        # başlangıç pozuna göre hizalanmış göreli sürüklenme (başlangıç yönelim hatası dahil değil)
        d_est = R[i0].T @ (p[i1] - p[i0])
        R_true0 = _true_R(flight, t[i0])
        d_ref = R_true0.T @ (ref[i1] - ref[i0])
        drift = np.linalg.norm(d_est - d_ref)
        sl = slice(i0, i1)
        rows.append({
            "t0": float(t[i0]), "dist": dist, "drift": float(drift),
            "drift_per_m": float(drift / dist),
            "n_corr": float(flight["lio_n_corr"][sl].mean()),
            "n_vert": float(flight["lio_n_vert"][sl].mean()),
            "n_edge": float(flight["lio_n_edge"][sl].mean()),
            "n_plane_scan": float(flight["lio_n_plane_scan"][sl].mean()),
            "normal_ev_min": float(flight["lio_normal_ev"][sl, 0].mean()),
            "pos_info_min": float(np.median(flight["lio_pos_info_ev"][sl, 0])),
            "rot_info_min": float(np.median(flight["lio_rot_info_ev"][sl, 0])),
            "speed": float(dist / seg),
        })
    return rows


def _true_R(flight, t):
    i = np.clip(np.searchsorted(flight["truth_t"], t - 1e-6), 0, len(flight["truth_t"]) - 1)
    return quat_to_rot(flight["truth_q"][i])
