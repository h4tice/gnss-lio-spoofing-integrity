"""MARS-LVIG (Li ve ark., IJRR 2024) ROS 1 bag okuyucusu.

Bag'den deneyde gereken akışları çıkarır ve sıkıştırılmış bir önbelleğe (npz) yazar:

* ``/livox/lidar``  (livox_ros_driver/CustomMsg) — taramalar, nokta başına zaman ofseti ile
  (hareket bozulması düzeltmesi için). Mesajın ham baytları doğrudan numpy ile çözülür.
* ``/livox/imu``    — LiDAR'ın dahili IMU'su (ivme g biriminde → m/s²'ye çevrilir)
* ``/ublox_driver/receiver_pvt`` — deney GNSS'i: ZED-F9P'nin kendi konum/hız çözümü (RTK'sız)
* ``/dji_osdk_ros/rtk_position``, ``rtk_velocity``, ``rtk_yaw`` — bağımsız RTK referansı

Tüm zamanlar mesaj başlığındaki (header.stamp) zamandır; ``time_report`` başlık ile bag
kayıt zamanı arasındaki farkları raporlar (saat tutarlılığı denetimi).
"""
import struct
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

G = 9.80665
LIDAR = "/livox/lidar"
TOPICS = {
    LIDAR: "lidar",
    "/livox/imu": "imu",
    "/ublox_driver/receiver_pvt": "pvt",
    "/dji_osdk_ros/rtk_position": "rtk",
    "/dji_osdk_ros/rtk_velocity": "rtk_vel",
    "/dji_osdk_ros/rtk_yaw": "rtk_yaw",
    "/dji_osdk_ros/rtk_info_position": "rtk_info",
}
POINT_DT = np.dtype([("off", "<u4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                     ("refl", "u1"), ("tag", "u1"), ("line", "u1")])


def parse_custom_msg(raw):
    """livox_ros_driver/CustomMsg (ROS 1 serileştirmesi) → (stamp, timebase_ns, noktalar)."""
    _, sec, nsec, flen = struct.unpack_from("<IIII", raw, 0)
    o = 16 + flen
    timebase, _ = struct.unpack_from("<QI", raw, o)
    o += 8 + 4 + 1 + 3
    (n,) = struct.unpack_from("<I", raw, o)
    pts = np.frombuffer(raw, dtype=POINT_DT, count=n, offset=o + 4)
    return sec + nsec * 1e-9, timebase, pts


def _header_t(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def lla_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """WGS-84 → yerel ENU (ECEF üzerinden, tam dönüşüm)."""
    a, e2 = 6378137.0, 6.69437999014e-3

    def ecef(la, lo, h):
        la, lo = np.radians(la), np.radians(lo)
        N = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
        return np.stack([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo),
                         (N * (1 - e2) + h) * np.sin(la)], axis=-1)

    d = ecef(lat, lon, alt) - ecef(lat0, lon0, alt0)
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    Rm = np.array([[-np.sin(lo0), np.cos(lo0), 0],
                   [-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)],
                   [np.cos(la0) * np.cos(lo0), np.cos(la0) * np.sin(lo0), np.sin(la0)]])
    return d @ Rm.T


def extract(bag_path, out_path, voxel=1.0, max_points=6000, min_range=2.0, log=print):
    """Bag'i bir kez okuyup önbellek npz dosyasına yazar."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    rng = np.random.default_rng(0)
    buf = {k: [] for k in TOPICS.values()}
    scans_xyz, scans_dt, scan_t = [], [], []
    lag = {k: [] for k in TOPICS.values()}
    with Reader(Path(bag_path)) as r:
        types = {}
        for c in r.connections:
            if c.topic in TOPICS and c.topic != LIDAR:
                types.update(get_types_from_msg(c.msgdef.data, c.msgtype))
        ts.register(types)
        conns = [c for c in r.connections if c.topic in TOPICS]
        for i, (c, tb, raw) in enumerate(r.messages(connections=conns)):
            key = TOPICS[c.topic]
            tbag = tb * 1e-9
            if key == "lidar":
                stamp, timebase, pts = parse_custom_msg(raw)
                t0 = timebase * 1e-9
                xyz = np.c_[pts["x"], pts["y"], pts["z"]].astype(np.float64)
                rn = np.linalg.norm(xyz, axis=1)
                tag = pts["tag"] & 0x30
                ok = (rn > min_range) & ((tag == 0x00) | (tag == 0x10))
                xyz, off = xyz[ok], pts["off"][ok] * 1e-9
                if len(xyz):
                    kk = np.floor(xyz / voxel).astype(np.int64)
                    _, first = np.unique(kk, axis=0, return_index=True)
                    if len(first) > max_points:
                        first = rng.choice(first, max_points, replace=False)
                    first = np.sort(first)
                    xyz, off = xyz[first], off[first]
                t_end = t0 + 0.1
                scan_t.append(t_end)
                scans_xyz.append(xyz.astype(np.float32))
                scans_dt.append((t_end - (t0 + off)).astype(np.float32))  # tarama sonuna kalan süre
                lag[key].append(tbag - t0)
                if len(scan_t) % 500 == 0:
                    log(f"  {len(scan_t)} tarama okundu")
                continue
            m = ts.deserialize_ros1(raw, c.msgtype)
            if key == "imu":
                t = _header_t(m)
                a, w = m.linear_acceleration, m.angular_velocity
                buf[key].append((t, a.x * G, a.y * G, a.z * G, w.x, w.y, w.z))
            elif key == "pvt":
                # GNSS zamanı (hafta, TOW) yerine bag zamanı: alıcı çözümü yayımlandığı an kullanılır
                t = tbag
                buf[key].append((t, m.latitude, m.longitude, m.altitude, m.vel_e, m.vel_n, -m.vel_d,
                                 m.h_acc, m.v_acc, m.vel_acc, m.num_sv, m.fix_type, m.carr_soln))
            elif key == "rtk":
                t = _header_t(m)
                buf[key].append((t, m.latitude, m.longitude, m.altitude))
            elif key == "rtk_vel":
                t = _header_t(m)
                v = m.vector
                buf[key].append((t, v.x, v.y, v.z))
            else:  # rtk_yaw, rtk_info (başlıksız): bag zamanı
                t = tbag
                buf[key].append((t, m.data))
            lag[key].append(tbag - t)

    arr = {k: np.array(v, float) for k, v in buf.items() if v}
    lens = np.array([len(s) for s in scans_xyz])
    report = {k: (float(np.median(v)), float(np.std(v))) for k, v in lag.items() if v}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, scan_t=np.array(scan_t), scan_len=lens,
                        scan_xyz=np.concatenate(scans_xyz), scan_dt=np.concatenate(scans_dt),
                        **{k: v for k, v in arr.items()})
    log(f"{len(scan_t)} tarama (ort. {lens.mean():.0f} nokta), "
        + ", ".join(f"{k}: {len(v)}" for k, v in arr.items()))
    log("bag kayıt zamanı − başlık zamanı [s] (medyan, std): "
        + ", ".join(f"{k} {m:+.3f}±{s:.3f}" for k, (m, s) in report.items()))
    return out_path


def load(path):
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    splits = np.cumsum(out.pop("scan_len"))[:-1]
    out["scans"] = np.split(out.pop("scan_xyz"), splits)
    out["scans_dt"] = np.split(out.pop("scan_dt"), splits)
    return out
