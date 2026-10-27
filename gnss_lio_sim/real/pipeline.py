"""MARS-LVIG dizileri üzerinde LIO taban hattı (İP1 gerçek veri doğrulaması).

Akış: bag → önbellek (mars_lvig.extract) → LIO (Livox IMU + deskew) → RTK'ya göre değerlendirme.

Saat ve çerçeve notları
* Livox LiDAR/IMU kendi saat damgalarını taşır; DJI RTK ve u-blox ROS saatindedir. Aradaki sabit
  fark, LIO hızı ile RTK hızının çapraz ilintisinden kestirilir (``clock_offset``).
* LIO kendi çerçevesinde başlar (yerçekimine hizalı, yaw = 0). ENU'ya hizalama, hareketin ilk
  ``align_duration`` saniyesinde (sağlıklı GNSS dönemi varsayımı) 4 serbestlik dereceli (yaw +
  öteleme) en küçük kareler ile yapılır; sonraki tüm hata bu hizalamadan sonra birikir.
"""
import json
import time
from pathlib import Path

import numpy as np

from ..geometry import quat_to_rot, rot_to_quat
from ..lio import LIO
from ..metrics import path_length, rpe, umeyama
from .mars_lvig import extract, lla_to_enu, load


def cache_path(cfg, seq):
    return Path(cfg.real.cache_dir) / f"{seq}.npz"


def ensure_cache(cfg, seq, log=print):
    cp = cache_path(cfg, seq)
    if not cp.exists():
        rc = cfg.real
        files = rc.get("files", {}).get(seq)  # MUN-FRL: dizi adı → bag / PPK / tek nokta dosyaları
        bag = Path(rc.bag_dir) / (files["bag"] if files else f"{seq}.bag")
        if not bag.exists():
            raise FileNotFoundError(f"{bag} bulunamadı (diziyi bu klasöre indirin)")
        log(f"{bag.name} okunuyor (bir kez yapılır)...")
        if rc.get("dataset", "mars_lvig") == "mun_frl":
            from .mun_frl import extract as extract_mun
            extract_mun(bag, cp, Path(rc.bag_dir) / files["ppk"], Path(rc.bag_dir) / files["single"],
                        rc.extract_voxel, rc.extract_max_points, rc.min_range, log,
                        extrinsic=(rc.lidar_to_imu_R, rc.lidar_to_imu_p) if "lidar_to_imu_R" in rc else None)
        else:
            extract(bag, cp, rc.extract_voxel, rc.extract_max_points, rc.min_range, log)
    return cp


def load_seq(cfg, seq, log=print):
    """Önbelleği okur; ``real.gnss_key`` verilmişse deney GNSS'i o akıştan alınır (MUN-FRL: pvt_rtk)."""
    d = load(ensure_cache(cfg, seq, log))
    key = cfg.real.get("gnss_key", "pvt")
    if key != "pvt":
        d["pvt"] = d[key]
    return d


def reference(d):
    """RTK referansı → ENU (kalkış noktası başlangıç), u-blox ölçümleri aynı ENU'da."""
    rtk = d["rtk"]
    # DJI bazı dizilerde (ör. HKairport_GNSS02) aynı konumu bir sonraki zaman damgasıyla yeniden
    # yayınlar (bayat mesaj); tekrarlar atılır, yoksa türev hızı ve saat farkı tahmini bozulur
    rtk = rtk[np.r_[True, np.any(np.diff(rtk[:, 1:4], axis=0) != 0, axis=1)]]
    lat0, lon0, alt0 = rtk[0, 1:4]
    ref = {"t": rtk[:, 0], "p": lla_to_enu(rtk[:, 1], rtk[:, 2], rtk[:, 3], lat0, lon0, alt0),
           "lla0": [float(lat0), float(lon0), float(alt0)]}
    # hız referansı RTK konumunun türevinden alınır: DJI rtk_velocity konusu konuma göre ~0,2 s
    # gecikmelidir (çapraz ilintiyle ölçüldü), zaman hizalamasını bozmasın
    ref["v_t"], ref["v"] = ref["t"], np.gradient(ref["p"], ref["t"], axis=0)
    if "pvt" not in d:  # deney GNSS'i kaydedilmemiş dizi: yarı-gerçek kip (bkz. build_flights)
        ref["synthetic_gnss"] = True
        return ref
    pvt = d["pvt"]
    ref["gnss_t"] = pvt[:, 0]
    ref["gnss_pos"] = lla_to_enu(pvt[:, 1], pvt[:, 2], pvt[:, 3], lat0, lon0, alt0)
    ref["gnss_vel"] = pvt[:, 4:7]
    ref["gnss_acc"] = pvt[:, 7:10]  # h_acc, v_acc, vel_acc (alıcının kendi tahmini)
    return ref


def rtk_dup_fraction(cfg, seq):
    """Referans kalite ölçütü: aynı konumu yeniden yayınlayan (bayat) RTK mesajlarının oranı."""
    r = load(ensure_cache(cfg, seq, lambda *a: None))["rtk"]
    return float(np.mean(np.all(np.diff(r[:, 1:4], axis=0) == 0, axis=1)))


def _init_attitude(acc_mean):
    """Yerçekimine hizalı başlangıç yönelimi (yaw = 0): R @ â = ẑ."""
    a = acc_mean / np.linalg.norm(acc_mean)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    s, c = np.linalg.norm(v), a @ z
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def run_lio(cfg, seq, log=print):
    d = load_seq(cfg, seq, log)
    ref = reference(d)
    rc = cfg.real
    imu = d["imu"]
    it, acc, gyr = imu[:, 0], imu[:, 1:4], imu[:, 4:7]
    t_lidar, scans, scans_dt = d["scan_t"], d["scans"], d["scans_dt"]
    t_lidar_off = np.asarray(rc.lidar_to_imu_t, float)

    # başlangıç: kalkıştan sonra start_altitude üstünde ilk an (Livox saatinde yaklaşık)
    k_alt = int(np.argmax(ref["p"][:, 2] > rc.start_altitude))
    t_start_ros = ref["t"][k_alt] + 2.0
    # iki saatin başlangıçları bag içinde ~0,3 s içinde örtüşür; başlangıç seçimi için yeterli
    valid = np.array([len(s) > 500 for s in scans])
    j0 = int(np.argmax(valid & (t_lidar > t_lidar[0] + (t_start_ros - ref["t"][0]))))
    t0 = t_lidar[j0]
    w = (it > t0 - rc.init_window) & (it <= t0)
    R0 = _init_attitude(acc[w].mean(0))
    bg0 = gyr[w].mean(0)
    lio = LIO(cfg.lio, cfg.imu, np.zeros(3), R0, np.zeros(3))
    lio.ins.bg = bg0.copy()

    # iniş: start_altitude altına son iniş (+2 s) sonrası yerde bekleme dönemi atılır (yalnız real.stop_at_landing)
    t_stop = np.inf
    if rc.get("stop_at_landing", False):
        k_land = len(ref["p"]) - 1 - int(np.argmax(ref["p"][::-1, 2] > rc.start_altitude))
        t_stop = ref["t"][k_land] + 2.0
    recs = []
    k = int(np.searchsorted(it, t0))
    tc = t0
    tic = time.perf_counter()
    for j in range(j0, len(t_lidar)):
        ts = t_lidar[j]
        if ts > t_stop:
            break
        while k < len(it) and it[k] <= ts:
            dt = it[k] - tc
            if dt > 0:
                lio.propagate(acc[k - 1], gyr[k - 1], dt)
                tc = it[k]
            k += 1
        if ts > tc:
            lio.propagate(acc[k - 1], gyr[k - 1], ts - tc)
            tc = ts
        if len(scans[j]) < 50:
            continue
        recs.append(lio.update(ts, scans[j].astype(float) + t_lidar_off, scans_dt[j]))
        if len(recs) % 500 == 0:
            log(f"  LIO {len(recs)} tarama, {time.perf_counter() - tic:.0f} s")
    wall = time.perf_counter() - tic
    log(f"LIO bitti: {len(recs)} tarama, {wall:.0f} s (ort. {1000 * wall / len(recs):.1f} ms/tarama)")
    return d, ref, recs


def clock_offset(t_lio, v_lio, ref, search=2.0, step=0.01):
    """Livox saati → ROS saati farkı: |v_LIO| ile |v_RTK| arasındaki en büyük ilinti."""
    sp_ref = np.linalg.norm(ref["v"], axis=1)
    sp = np.linalg.norm(v_lio, axis=1)
    best = (-2.0, 0.0)
    # iki saat da Unix zamanındadır (bag içinde fark ~0,3 s), yalnız küçük fark aranır
    for off in np.arange(-search, search + step / 2, step):
        tt = t_lio + off
        ok = (tt > ref["v_t"][0]) & (tt < ref["v_t"][-1])
        x = np.interp(tt[ok], ref["v_t"], sp_ref)
        c = np.corrcoef(x, sp[ok])[0, 1]
        best = max(best, (c, off))
    return best[1], best[0]


def align_4dof(src, dst):
    """src → dst: yalnız yaw + öteleme (yerçekimi yönü gözlenebilir olduğundan roll/pitch hizalanmaz)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src[:, :2] - ms[:2], dst[:, :2] - md[:2]
    yaw = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R, md - R @ ms


def evaluate_lio(recs, ref, cfg):
    rc = cfg.real
    t = np.array([r["t"] for r in recs])
    p = np.array([r["p"] for r in recs])
    v = np.array([r["v"] for r in recs])
    off, corr = clock_offset(t, v, ref)
    tr = t + off  # ROS saatinde LIO zamanları
    ok = (tr > ref["t"][0]) & (tr < ref["t"][-1])
    tr, p = tr[ok], p[ok]
    gt = np.c_[[np.interp(tr, ref["t"], ref["p"][:, i]) for i in range(3)]].T
    sp = np.linalg.norm(np.gradient(gt, tr, axis=0)[:, :2], axis=1)
    moving = np.where(sp > 1.0)[0]
    m0 = moving[0]
    win = (tr >= tr[m0]) & (tr <= tr[m0] + rc.align_duration)
    R4, t4 = align_4dof(p[win], gt[win])
    p_al = p @ R4.T + t4
    Rg, tg, _ = umeyama(p, gt)
    p_gl = p @ Rg.T + tg
    err = np.linalg.norm(p_al - gt, axis=1)
    after = tr >= tr[m0] + rc.align_duration
    # 60 s'lik kayan pencerelerde GNSS'siz sürüklenme (pencere başında konum sıfırlanır)
    drifts = []
    for ts in np.arange(tr[m0], tr[-1] - 60.0, 10.0):
        a, b = np.searchsorted(tr, [ts, ts + 60.0])
        if b >= len(tr) or b - a < 100:
            continue
        de = (p_al[b] - p_al[a]) - (gt[b] - gt[a])
        L = path_length(gt[a:b + 1])
        if L > 50:
            drifts.append((ts - tr[0], np.linalg.norm(de), 100 * np.linalg.norm(de) / L, L))
    drifts = np.array(drifts)
    res = {
        "clock_offset": float(off), "clock_corr": float(corr),
        "n_scans": int(len(t)), "duration": float(tr[-1] - tr[0]),
        "path_length": path_length(gt),
        "ate_global_aligned": float(np.sqrt(np.mean(np.sum((p_gl - gt) ** 2, 1)))),
        "ate_start_aligned": float(np.sqrt(np.mean(err ** 2))),
        "final_err_start_aligned": float(err[-1]),
        "final_err_pct": float(100 * err[-1] / path_length(gt[m0:])),
        "rpe_1s": rpe(tr, p_gl, gt, 1.0),
        "drift60_pct_mean": float(drifts[:, 2].mean()) if len(drifts) else float("nan"),
        "drift60_pct_max": float(drifts[:, 2].max()) if len(drifts) else float("nan"),
        "drift60_m_mean": float(drifts[:, 1].mean()) if len(drifts) else float("nan"),
        "proc_ms_mean": float(1000 * np.mean([r["proc_time"] for r in recs])),
        "proc_ms_p95": float(1000 * np.percentile([r["proc_time"] for r in recs], 95)),
        "err_after_align_mean": float(err[after].mean()) if after.any() else float("nan"),
    }
    series = {"t": tr, "p_lio": p_al, "p_gl": p_gl, "gt": gt, "R4": R4, "t4": t4, "drifts": drifts,
              "moving_start": float(tr[m0])}
    return res, series


def gnss_latency(ref, search=1.0, step=0.01):
    """u-blox çözümünün yayın gecikmesi: GNSS hızı ile RTK hızını en iyi örtüştüren zaman kayması."""
    best = (np.inf, 0.0)
    for lag in np.arange(-search, search + step / 2, step):
        tt = ref["gnss_t"] + lag
        ok = (tt > ref["v_t"][0]) & (tt < ref["v_t"][-1])
        vr = np.c_[[np.interp(tt[ok], ref["v_t"], ref["v"][:, i]) for i in range(2)]].T
        e = np.sqrt(np.mean(np.sum((ref["gnss_vel"][ok, :2] - vr) ** 2, axis=1)))
        best = min(best, (e, lag))
    return best[1], best[0]


def _interp(t, tt, x):
    return np.c_[[np.interp(t, tt, x[:, i]) for i in range(x.shape[1])]].T


def build_flights(cfg, seq, lio_dir, out_dir, log=print):
    """Gerçek diziyi benzetimin uçuş kaydı biçimine dönüştürür (ortak 200 Hz zaman ızgarası).

    * Zaman tabanı ROS saati; Livox akışları ``clock_offset`` ile, u-blox ``gnss_latency`` ile düzeltilir.
    * LIO çerçevesi ENU'ya, hareketin ilk ``align_duration`` saniyesindeki **u-blox** konumlarıyla
      (sağlıklı GNSS dönemi; RTK kullanılmaz) 4 SD hizalanır — gerçek bir sistemin yapabileceği gibi.
    * Referans (RTK), aynı pencerede ölçülen sabit u-blox − RTK farkı kadar kaydırılarak GNSS
      datumunda ifade edilir (aldatmadan bağımsız mutlak datum farkı değerlendirmeye karışmasın).
    * Her dizi, farklı saldırı başlangıçları ve yönleriyle ``cfg.real.n_onsets`` kayda çoğaltılır.
    """
    import zlib

    rc = cfg.real
    d = load_seq(cfg, seq)
    ref = reference(d)
    z = np.load(Path(lio_dir) / f"{seq}_lio.npz")
    lres = json.loads(str(z["meta"]))
    off = lres["clock_offset"]
    synthetic = ref.get("synthetic_gnss", False)
    lag, lag_rms = (0.0, 0.0) if synthetic else gnss_latency(ref)
    log(f"[{seq}] Livox→ROS saat farkı {off:+.3f} s, u-blox gecikmesi {lag:+.2f} s (hız RMS {lag_rms:.3f} m/s)")

    # --- ortak ızgara: ilk LIO taraması t0, 200 Hz; taramalar her 20 örnekte bir
    t_l = z["lio_t"] + off
    t0 = t_l[0]
    dt = 1.0 / cfg.imu.rate
    scan_every = int(round(cfg.imu.rate / cfg.lidar.rate))
    gnss_every = int(round(cfg.imu.rate / cfg.gnss.rate))
    n_scan = np.round((t_l - t0) * cfg.lidar.rate).astype(int)
    snap = np.abs((t_l - t0) - n_scan / cfg.lidar.rate)
    keep = np.r_[True, np.diff(n_scan) > 0]
    log(f"  tarama zamanlarının ızgaraya uzaklığı: en çok {1000 * snap.max():.1f} ms")
    K = int(n_scan[-1]) * scan_every
    tg = t0 + np.arange(K + 1) * dt

    imu = d["imu"]
    ti = imu[:, 0] + off
    imu_acc, imu_gyro = _interp(tg, ti, imu[:, 1:4]), _interp(tg, ti, imu[:, 4:7])

    lio_t = tg[n_scan[keep] * scan_every]
    Rl = quat_to_rot(z["lio_q"][keep])
    pl = z["lio_p"][keep]

    # --- GNSS (u-blox) 5 Hz çağlar: en yakın ölçüm
    ge = tg[::gnss_every]
    if synthetic:
        gnss_t, gnss_pos, gnss_vel, donor = replay_gnss(cfg, lio_dir, ge, ref)
        tgn, g_all = gnss_t, gnss_pos
        log(f"  deney GNSS'i yok: RTK + {donor} uçuşunda ölçülmüş u-blox hata dizisi (yarı-gerçek)")
    else:
        tgn, g_all, donor = ref["gnss_t"] + lag, ref["gnss_pos"], None
        j = np.clip(np.searchsorted(tgn, ge), 1, len(tgn) - 1)
        if rc.get("gnss_resample", "nearest") == "interp":
            # GNSS çağları ızgaraya denk gelmiyorsa (MUN-FRL: tarama hızı 10,08 Hz) iki komşu ölçüm arasında
            # doğrusal ara değerleme; komşular arası boşluk > 0,3 s ise o çağ kullanılmaz (kesinti korunur)
            okg = (ge >= tgn[0]) & (ge <= tgn[-1]) & (tgn[j] - tgn[j - 1] < 0.3)
            gnss_t = ge[okg]
            gnss_pos, gnss_vel = _interp(gnss_t, tgn, ref["gnss_pos"]), _interp(gnss_t, tgn, ref["gnss_vel"])
        else:
            j -= (ge - tgn[j - 1]) < (tgn[j] - ge)
            okg = np.abs(tgn[j] - ge) < 0.06
            gnss_t, gnss_pos, gnss_vel = ge[okg], ref["gnss_pos"][j[okg]], ref["gnss_vel"][j[okg]]

    # --- referans 10 Hz (doğrusal ara değerleme) ve hizalama penceresi
    rtk_p = _interp(lio_t, ref["t"], ref["p"])
    sp = np.linalg.norm(np.gradient(rtk_p, lio_t, axis=0)[:, :2], axis=1)
    m0 = lio_t[np.argmax(sp > 1.0)]
    win = (lio_t >= m0) & (lio_t <= m0 + rc.align_duration)
    g_win = _interp(lio_t[win], tgn, g_all)
    R4, t4 = align_4dof(pl[win], g_win)
    lio_p = pl @ R4.T + t4
    lio_R = np.einsum("ij,njk->nik", R4, Rl)
    datum = (g_win - rtk_p[win]).mean(0)
    rtk_p = rtk_p + datum
    log(f"  hizalama: yaw {np.degrees(np.arctan2(R4[1, 0], R4[0, 0])):+.1f}°, "
        f"u-blox − RTK datum farkı (E,N,U) {datum.round(2)} m")

    # GNSS hata modeli denetimi (yalnız bilgi amaçlı)
    e = gnss_pos - _interp(gnss_t, lio_t, rtk_p)
    wn = np.diff(e, axis=0).std(0) / np.sqrt(2)
    log(f"  u-blox hatası (datum sonrası) std E/N/U {e.std(0).round(2)} m, beyaz bileşen ≈ {wn.round(2)} m")
    if not synthetic:
        # bu uçuşun ölçülmüş u-blox hata dizisi: deney GNSS'i olmayan dizilerde yeniden kullanılır
        ev = gnss_vel - _interp(gnss_t, ref["v_t"], ref["v"])
        np.savez_compressed(Path(lio_dir) / f"{seq}_gnss_err.npz", t=gnss_t - gnss_t[0], e=e, ev=ev)

    rel = lambda x: x - t0  # noqa: E731
    base = dict(
        imu_t=rel(tg), imu_acc=imu_acc, imu_gyro=imu_gyro,
        imu_ba=np.zeros_like(imu_acc[:1]), imu_bg=np.zeros_like(imu_acc[:1]),
        gnss_t=rel(gnss_t), gnss_pos=gnss_pos, gnss_vel=gnss_vel,
        rtk_t=rel(lio_t), rtk_p=rtk_p,
        truth_t=rel(lio_t), truth_p=rtk_p, truth_v=np.gradient(rtk_p, lio_t, axis=0),
        truth_q=rot_to_quat(lio_R),  # ENU'ya hizalı LIO yönelimi (yalnız başlangıç ve bölüt ekseni için)
        lio_t=rel(lio_t), lio_p=lio_p, lio_q=rot_to_quat(lio_R),
    )
    for k in z.files:
        if k.startswith("lio_") and k not in ("lio_t", "lio_p", "lio_q", "lio_R", "lio_v"):
            base[k] = z[k][keep]
    base["lio_v"] = z["lio_v"][keep] @ R4.T

    dur = rel(lio_t)[-1]
    span = rc.attack_span
    warmup = rel(m0) + rc.align_duration
    # görev sonu: hizalamadan sonra ilk hızlı yerinde dönüş (|ω| > spin_rate) ya da kayıt sonu
    w_norm = np.linalg.norm(imu_gyro, axis=1)
    spin = np.where((rel(tg) > warmup) & (w_norm > rc.spin_rate))[0]
    mission_end = float(rel(tg[spin[0]]) - 2.0) if len(spin) else float(dur)
    first = warmup + 10.0
    last = min(dur, mission_end) - cfg.flight.gnss_denied_duration - span
    log(f"  işler durum {warmup:.0f} s, görev sonu {mission_end:.0f} s (sonrası yerinde dönüş; "
        f"değerlendirme dışı, yalnız far_full'da)")
    onsets = np.linspace(first, last, rc.n_onsets)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plen = path_length(rtk_p)
    speed = float(np.median(sp[sp > 1.0]))
    log(f"  süre {dur:.0f} s, yol {plen:.0f} m, saldırı başlangıçları {onsets.round(0)} s")
    for i, onset in enumerate(onsets):
        fid = f"{seq}_o{i}"
        seed = (zlib.crc32(fid.encode()) + cfg.seed) % (2 ** 31)
        rng = np.random.default_rng(seed)
        a = rng.normal(size=2)
        meta = {"id": fid, "env": rc.env_label.get(seq, seq), "traj": seq, "speed": round(speed, 1),
                "rep": i, "seed": int(seed), "onset": float(onset), "duration": float(dur),
                "denied": cfg.flight.gnss_denied_duration, "attack_dir": np.r_[a / np.linalg.norm(a), 0].tolist(),
                "path_length": plen, "clock_offset": off, "gnss_latency": float(lag),
                "warmup": float(warmup), "mission_end": mission_end,
                "datum_offset": datum.tolist(), "source": "MARS-LVIG",
                "gnss_source": f"replay:{donor}" if synthetic else "u-blox"}
        np.savez_compressed(out / f"{fid}.npz", **base, meta=json.dumps(meta))
    return [out / f"{seq}_o{i}.npz" for i in range(len(onsets))]


def replay_gnss(cfg, lio_dir, ge, ref):
    """Yarı-gerçek GNSS: RTK konum/hızı + aynı kampanyada başka bir uçuşta ölçülmüş u-blox hata dizisi
    (``real.gnss_donor``). Hata dizisi kısaysa zamanda aynalanarak (süreklilik korunur) uzatılır."""
    donor = cfg.real.gnss_donor
    f = Path(lio_dir) / f"{donor}_gnss_err.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} yok: önce {donor} dizisini real-eval ile işleyin")
    z = np.load(f)
    t, e, ev = z["t"], z["e"], z["ev"]
    tt = ge - ge[0]
    while t[-1] < tt[-1]:
        t = np.r_[t, 2 * t[-1] - t[-2::-1]]
        e, ev = np.r_[e, e[-2::-1]], np.r_[ev, ev[-2::-1]]
    # hata dizisinin ilk anı, uçuşun hizalama öncesi dönemine denk gelir (datum farkı ayrıca giderilir)
    pos = _interp(ge, ref["t"], ref["p"]) + _interp(tt, t, e)
    vel = _interp(ge, ref["v_t"], ref["v"]) + _interp(tt, t, ev)
    return ge, pos, vel, donor


def save_lio(out_dir, seq, recs, res, series, ref):
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    lio_arr = {f"lio_{k}": np.array([r[k] for r in recs]) for k in recs[0] if k != "R"}
    lio_arr["lio_q"] = rot_to_quat(np.array([r["R"] for r in recs]))
    np.savez_compressed(out / f"{seq}_lio.npz", **lio_arr,
                        **{f"s_{k}": v for k, v in series.items()},
                        meta=json.dumps(res))
    (out / f"{seq}_lio.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(out / "figures" / f"{seq}_lio.png", seq, series, ref)


def _plot(path, seq, s, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    if "gnss_pos" in ref:  # yarı-gerçek dizilerde kaydedilmiş u-blox yok
        a.plot(ref["gnss_pos"][:, 0], ref["gnss_pos"][:, 1], ".", ms=1.2, color="#eda100", alpha=0.6,
               label="u-blox GNSS (RTK'sız)")
    a.plot(s["gt"][:, 0], s["gt"][:, 1], color="#555555", lw=2, label="Referans (DJI RTK)")
    a.plot(s["p_lio"][:, 0], s["p_lio"][:, 1], color="#008300", lw=1.4,
           label="Yalnız LIO (ilk 60 s'de hizalı)")
    a.set_aspect("equal")
    a.set_xlabel("Doğu [m]")
    a.set_ylabel("Kuzey [m]")
    a.set_title(f"{seq} — üstten görünüm")
    a.legend(fontsize=8, loc="best")
    a.grid(alpha=0.3)
    b = ax[1]
    e = np.linalg.norm(s["p_lio"] - s["gt"], axis=1)
    eh = np.linalg.norm((s["p_lio"] - s["gt"])[:, :2], axis=1)
    tt = s["t"] - s["t"][0]
    b.plot(tt, e, color="#008300", lw=1.4, label="3B hata")
    b.plot(tt, eh, color="#2a78d6", lw=1.2, label="yatay hata")
    b.axvspan(s["moving_start"] - s["t"][0], s["moving_start"] - s["t"][0] + 60, color="#999999", alpha=0.15,
              label="hizalama penceresi")
    b.set_xlabel("Zaman [s]")
    b.set_ylabel("Konum hatası [m]")
    b.set_title("Yalnız LIO hatası (GNSS kullanılmadan)")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
