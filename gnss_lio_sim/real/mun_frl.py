"""MUN-FRL (Thalagala ve ark., IJRR 2024) ROS 1 bag okuyucusu — ikinci açık gerçek veri seti.

DJI M600 + Velodyne VLP-16 (aşağı bakan) + Xsens MTi-30 IMU + simpleRTK2B GNSS. Önbellek biçimi
``mars_lvig.extract`` ile aynıdır (pipeline değişmeden kullanılır):

* ``/velodyne_points`` (PointCloud2; x, y, z, intensity, ring, time) — nokta zamanı başlık zamanına göredir;
  başlık bazı uçuşlarda taramanın sonu (quarry1: −0,08…0 s), bazılarında başıdır (lighthouse, quarry2:
  0…0,09 s). Mutlak nokta zamanı = başlık + time; tarama zamanı en geç noktanınkidir. Noktalar IMU
  çerçevesine (``imu_up``: z yukarı) dönüştürülür
* ``/imu/data`` — Xsens 400 Hz (z aşağı, ``imu_link``) → ``imu_up`` çerçevesine çevrilir (y ekseni etrafında
  180°). Başlık zamanları USB aktarımı nedeniyle kümelenir (0-30 ms gecikme); zaman damgası, sensörün kendi
  saati (``/imu/time_ref``) ile başlık zamanının alt zarfına doğrusal uydurmadan yeniden kurulur (``imu_time``)
* deney GNSS'i (iki rejim, ikisi de önbelleğe yazılır):
    ``pvt``     : drone alıcısının ham gözlemlerinden hesaplanmış **RTK'sız tek nokta** çözümü
                  (veri setinin ``<dizi>.pos`` dosyası, RTKLIB "Single")
    ``pvt_rtk`` : uçuş sırasında yayımlanan **RTK-sabit** çözüm (``/fix``, GGA kalite 4)
  GNSS hızı yayımlanmadığından konumun merkezi farkından türetilir (0,4 s taban).
* referans ``rtk``: PPK çözümü (``<dizi>_ppk.pos``, RTKLIB Kinematic, bağımsız baz istasyonu ile)

Zamanlar: ``.pos`` dosyaları GPS zamanındadır (UTC = GPST − 18 s, 2022). Bag başlık zamanları kayıt
bilgisayarının saatindedir ve tam uçuş bag'lerinde yanlıştır (ör. quarry1: 2018); bag saati UTC'ye
``bag_to_utc`` ile çevrilir (NMEA GGA günün saati ile kaba, RTK-sabit /fix ile PPK konumlarının
karşılaştırılmasıyla ince hizalama) ve tüm bag akışları (IMU, LiDAR, /fix) aynı farkla kaydırılır.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from ..geometry import quat_to_rot
from .mars_lvig import _header_t, lla_to_enu

GPS_UTC_LEAP = 18.0
LIDAR, IMU, FIX, TF, NMEA, IMU_REF = ("/velodyne_points", "/imu/data", "/fix", "/tf_static", "/nmea_sentence",
                                     "/imu/time_ref")
R_UP = np.diag([-1.0, 1.0, -1.0])  # imu_link → imu_up (tf_static: y ekseni etrafında 180°)
PC2_FMT = {2: "u1", 4: "<u2", 6: "<u4", 7: "<f4", 8: "<f8"}


def read_pos(path):
    """RTKLIB .pos → (UTC unix zamanı, [lat, lon, h], Q, [sdn, sde, sdu])."""
    d = pd.read_csv(path, comment="%", sep=r"\s+", header=None)
    t = (pd.to_datetime(d[0] + " " + d[1]) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    return t - GPS_UTC_LEAP, d[[2, 3, 4]].to_numpy(float), d[5].to_numpy(int), d[[7, 8, 9]].to_numpy(float)


def _vel_from_pos(t, lla, base=0.4):
    """Konumun merkezi farkıyla ENU hızı (alıcı hızı yayımlanmadığı için)."""
    p = lla_to_enu(lla[:, 0], lla[:, 1], lla[:, 2], *lla[0])
    k = max(1, int(round(base / 2 / np.median(np.diff(t)))))
    v = np.full_like(p, np.nan)
    v[k:-k] = (p[2 * k:] - p[:-2 * k]) / (t[2 * k:] - t[:-2 * k])[:, None]
    v[:k], v[-k:] = v[k], v[-k - 1]
    return v


def imu_time(t_hdr, t_sensor, nbins=50):
    """Sensör saati → bag saati: başlık − sensör farkının alt zarfına (aktarım gecikmesi yalnız pozitif)
    doğrusal uydurma (iki saat arasındaki hız farkı ~40 ppm)."""
    d = t_hdr - t_sensor
    idx = np.array_split(np.arange(len(d)), nbins)
    x = np.array([t_sensor[i].mean() for i in idx])
    y = np.array([np.percentile(d[i], 1) for i in idx])
    b, a = np.polyfit(x - x[0], y, 1)
    return t_sensor + a + b * (t_sensor - x[0])


def bag_to_utc(gga, fix, ppk_t, ppk_lla, log=print):
    """Bag saati → UTC farkı [s]. gga: [(bag t, GGA cümlesi)], fix: [bag t, lat, lon, h], PPK: UTC."""
    day0 = np.floor(ppk_t[0] / 86400.0) * 86400.0
    c = []
    for t, s in gga:
        f = s.split(",")
        if len(f) > 1 and len(f[1]) >= 6:
            tod = int(f[1][:2]) * 3600 + int(f[1][2:4]) * 60 + float(f[1][4:])
            c.append(day0 + tod - t)
    if not c:
        raise RuntimeError("NMEA GGA cümlesi bulunamadı: bag saati UTC'ye çevrilemiyor")
    coarse = float(np.median(c))
    # ince: RTK-sabit /fix ile PPK yatay konum farkının ortancasını en küçükleyen kayma (hareketli dönem)
    p_ref = lla_to_enu(ppk_lla[:, 0], ppk_lla[:, 1], ppk_lla[:, 2], *ppk_lla[0])
    p_fix = lla_to_enu(fix[:, 1], fix[:, 2], fix[:, 3], *ppk_lla[0])
    best = (np.inf, 0.0)
    for dt in np.arange(-1.0, 1.0 + 1e-9, 0.005):
        tt = fix[:, 0] + coarse + dt
        ok = (tt > ppk_t[0]) & (tt < ppk_t[-1])
        if ok.sum() < 50:
            continue
        e = [np.interp(tt[ok], ppk_t, p_ref[:, i]) - p_fix[ok, i] for i in range(2)]
        best = min(best, (float(np.median(np.hypot(*e))), dt))
    log(f"  bag saati → UTC: kaba {coarse:.3f} s, ince {best[1]:+.3f} s (/fix-PPK yatay fark ortancası "
        f"{100 * best[0]:.1f} cm)")
    return coarse + best[1]


def _tf(transforms):
    T = {}
    for tr in transforms:
        q, p = tr.transform.rotation, tr.transform.translation
        T[(tr.header.frame_id, tr.child_frame_id)] = (quat_to_rot(np.array([q.w, q.x, q.y, q.z])),
                                                      np.array([p.x, p.y, p.z]))
    return T


def extract(bag_path, out_path, ppk_path, single_path, voxel=1.0, max_points=6000, min_range=2.0, log=print,
            extrinsic=None):
    """``extrinsic``: (R, p) velodyne → imu_up; tam uçuş bag'lerinde ``/tf_static`` yoktur, örnek bag'den alınır."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    rng = np.random.default_rng(0)
    imu, imu_ns, fix, gga, imu_ref = [], [], [], [], {}
    scans_xyz, scans_dt, scan_t = [], [], []
    ext = None
    with Reader(Path(bag_path)) as r:
        tf_conns = [c for c in r.connections if c.topic == TF]
        types = {}
        for c in r.connections:
            if c.topic not in (TF, NMEA):
                continue
            types.update(get_types_from_msg(c.msgdef.data, c.msgtype))
        ts.register(types)
        if extrinsic is not None:
            ext = tuple(np.asarray(x, float) for x in extrinsic)
        # dış parametreler: velodyne ve imu_link, camera çerçevesine göre verilir
        for c, _, raw in (r.messages(connections=tf_conns) if tf_conns and ext is None else []):
            T = _tf(ts.deserialize_ros1(raw, c.msgtype).transforms)
            if ("camera", "velodyne") in T and ("camera", "imu_link") in T:
                Rcv, pcv = T[("camera", "velodyne")]
                Rci, pci = T[("camera", "imu_link")]
                R_iv = R_UP @ Rci.T @ Rcv                     # velodyne → imu_up
                p_iv = R_UP @ (Rci.T @ (pcv - pci))
                ext = (R_iv, p_iv)
                break
        if ext is None:
            raise RuntimeError("tf_static içinde camera→velodyne / camera→imu_link bulunamadı")
        log(f"  LiDAR→IMU: öteleme {ext[1].round(3)} m")
        conns = [c for c in r.connections if c.topic in (LIDAR, IMU, FIX, NMEA, IMU_REF)]
        for c, _, raw in r.messages(connections=conns):
            m = ts.deserialize_ros1(raw, c.msgtype)
            if c.topic == IMU:
                a = R_UP @ np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
                w = R_UP @ np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
                imu.append((_header_t(m), *a, *w))
                imu_ns.append(m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)
            elif c.topic == IMU_REF:
                imu_ref[m.header.stamp.sec * 10**9 + m.header.stamp.nanosec] = m.time_ref.sec + m.time_ref.nanosec * 1e-9
            elif c.topic == NMEA:
                if "GGA" in m.sentence[:7]:
                    gga.append((_header_t(m), m.sentence))
            elif c.topic == FIX:
                fix.append((_header_t(m), m.latitude, m.longitude, m.altitude, m.status.status))
            else:
                dt = np.dtype({"names": [f.name for f in m.fields],
                               "formats": [PC2_FMT[f.datatype] for f in m.fields],
                               "offsets": [f.offset for f in m.fields], "itemsize": m.point_step})
                pts = np.frombuffer(bytes(m.data), dtype=dt)
                xyz = np.c_[pts["x"], pts["y"], pts["z"]].astype(np.float64)
                off = pts["time"].astype(np.float64)
                ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > min_range)
                xyz, off = xyz[ok], off[ok]
                if len(xyz):
                    kk = np.floor(xyz / voxel).astype(np.int64)
                    _, first = np.unique(kk, axis=0, return_index=True)
                    if len(first) > max_points:
                        first = rng.choice(first, max_points, replace=False)
                    first = np.sort(first)
                    xyz, off = xyz[first], off[first]
                xyz = xyz @ ext[0].T + ext[1]
                t_end = _header_t(m) + max(float(off.max()) if len(off) else 0.0, 0.0)  # taramanın sonu
                scan_t.append(t_end)
                scans_xyz.append(xyz.astype(np.float32))
                scans_dt.append((t_end - _header_t(m) - off).astype(np.float32))
                if len(scan_t) % 500 == 0:
                    log(f"  {len(scan_t)} tarama okundu")

    # referans: PPK (yalnız sabit çözüm, Q = 1)
    t, lla, q, _ = read_pos(ppk_path)
    out = {"rtk": np.c_[t, lla][q == 1]}
    log(f"  PPK: {len(t)} çağ, sabit çözüm %{100 * np.mean(q == 1):.1f}")
    f = np.array(fix, float)
    off = bag_to_utc(gga, f, out["rtk"][:, 0], out["rtk"][:, 1:4], log)
    f[:, 0] += off
    imu = np.array(imu, float)
    key = imu_ns
    has = np.array([k in imu_ref for k in key])
    if has.mean() > 0.99:  # sensör saati: eşi olmayan birkaç ölçüm atılır
        imu = imu[has]
        imu[:, 0] = imu_time(imu[:, 0], np.array([imu_ref[k] for k, h in zip(key, has) if h]))
        imu = imu[np.r_[True, np.diff(imu[:, 0]) > 0]]
        log(f"  IMU zamanı sensör saatinden kuruldu ({len(imu)} ölçüm, {1 / np.median(np.diff(imu[:, 0])):.0f} Hz)")
    out["imu"] = imu[:, :7]
    out["imu"][:, 0] += off
    scan_t = np.array(scan_t) + off
    # deney GNSS'i 1: RTK'sız tek nokta çözümü
    t, lla, q, sd = read_pos(single_path)
    v = _vel_from_pos(t, lla)
    hacc = np.hypot(sd[:, 0], sd[:, 1])
    out["pvt"] = np.c_[t, lla, v, hacc, sd[:, 2], np.full(len(t), np.nan), np.full(len(t), np.nan), q,
                       np.zeros(len(t))]
    # deney GNSS'i 2: uçuşta yayımlanan RTK-sabit çözüm
    v = _vel_from_pos(f[:, 0], f[:, 1:4])
    n = len(f)
    out["pvt_rtk"] = np.c_[f[:, 0:4], v, np.full((n, 3), np.nan), np.full(n, np.nan), f[:, 4], np.full(n, 2.0)]
    log(f"  tek nokta: {len(out['pvt'])} çağ; /fix: {n} çağ (durum {np.unique(f[:, 4])})")
    lens = np.array([len(s) for s in scans_xyz])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, scan_t=scan_t, scan_len=lens, scan_xyz=np.concatenate(scans_xyz),
                        scan_dt=np.concatenate(scans_dt), **out)
    log(f"{len(scan_t)} tarama (ort. {lens.mean():.0f} nokta), imu: {len(imu)}")
    return out_path
