"""Uçuş kaydını ROS 2 bag'e (ROS kurulumu gerektirmeden, `rosbags` ile) dışa aktarır.

Böylece benzetim verisi, ileride gerçek donanım/ROS 2 ortamında FAST-LIO2 / Point-LIO
gibi resmi gerçeklemelerle aynı biçimde işlenebilir ve veri seti olarak paylaşılabilir.

Konular:
  /imu                    sensor_msgs/Imu          (200 Hz)
  /lidar/points           sensor_msgs/PointCloud2  (10 Hz, gövde/LiDAR çerçevesi)
  /gnss/fix               sensor_msgs/NavSatFix    (5 Hz, saldırı enjekte edilmiş)
  /gnss/fix_clean         sensor_msgs/NavSatFix    (5 Hz, sağlıklı)
  /ground_truth/odom      nav_msgs/Odometry        (10 Hz, RTK referansı)
  /lio/odom               nav_msgs/Odometry        (10 Hz, yalnız-LIO çıktısı)
"""
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from .geometry import quat_to_rot

T0_NS = 1_780_000_000 * 10 ** 9  # sabit epoch (2026)
EARTH_R = 6_378_137.0


def _enu_to_lla(p, lat0, lon0, alt0):
    lat = lat0 + np.rad2deg(p[:, 1] / EARTH_R)
    lon = lon0 + np.rad2deg(p[:, 0] / (EARTH_R * np.cos(np.deg2rad(lat0))))
    return lat, lon, alt0 + p[:, 2]


def export_bag(flight, scans, gnss_att, out_path, cfg):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    Header, Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
    Vec3, Quat, Point = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Point"]
    Imu, PC2, PF = T["sensor_msgs/msg/Imu"], T["sensor_msgs/msg/PointCloud2"], T["sensor_msgs/msg/PointField"]
    Fix, FixSt = T["sensor_msgs/msg/NavSatFix"], T["sensor_msgs/msg/NavSatStatus"]
    Odom, Pose, PoseC = T["nav_msgs/msg/Odometry"], T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/PoseWithCovariance"]
    Twist, TwistC = T["geometry_msgs/msg/Twist"], T["geometry_msgs/msg/TwistWithCovariance"]

    def hdr(t, frame):
        ns = T0_NS + int(round(t * 1e9))
        return Header(stamp=Time(sec=ns // 10 ** 9, nanosec=ns % 10 ** 9), frame_id=frame), ns

    def quat(q):  # [w,x,y,z] -> geometry_msgs
        return Quat(x=float(q[1]), y=float(q[2]), z=float(q[3]), w=float(q[0]))

    out_path = Path(out_path)
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    msgs = []
    for k, t in enumerate(flight["imu_t"]):
        h, ns = hdr(t, "imu_link")
        a, w = flight["imu_acc"][k], flight["imu_gyro"][k]
        m = Imu(header=h, orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=np.r_[-1.0, np.zeros(8)],
                angular_velocity=Vec3(x=float(w[0]), y=float(w[1]), z=float(w[2])),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vec3(x=float(a[0]), y=float(a[1]), z=float(a[2])),
                linear_acceleration_covariance=np.zeros(9))
        msgs.append(("/imu", Imu, ns, m))
    fields = [PF(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(("x", "y", "z", "intensity"))]
    for t, pts in zip(flight["lio_t"], scans):
        h, ns = hdr(t, "lidar_link")
        arr = np.c_[pts, np.full(len(pts), 100.0)].astype(np.float32)
        m = PC2(header=h, height=1, width=len(arr), fields=fields, is_bigendian=False, point_step=16,
                row_step=16 * len(arr), data=arr.view(np.uint8).ravel(), is_dense=True)
        msgs.append(("/lidar/points", PC2, ns, m))
    g = cfg.gnss
    cov = np.diag([g.sigma_h ** 2, g.sigma_h ** 2, g.sigma_v ** 2]).ravel()
    for topic, pos, valid in (("/gnss/fix", gnss_att["pos"], gnss_att["valid"]),
                              ("/gnss/fix_clean", flight["gnss_pos"], np.ones(len(flight["gnss_t"]), bool))):
        lat, lon, alt = _enu_to_lla(pos, g.ref_lat, g.ref_lon, g.ref_alt)
        for k, t in enumerate(flight["gnss_t"]):
            if not valid[k]:
                continue
            h, ns = hdr(t, "gnss_link")
            m = Fix(header=h, status=FixSt(status=0, service=1), latitude=float(lat[k]), longitude=float(lon[k]),
                    altitude=float(alt[k]), position_covariance=cov, position_covariance_type=2)
            msgs.append((topic, Fix, ns, m))
    rtk_idx = np.searchsorted(flight["truth_t"], flight["rtk_t"] - 1e-6)
    for topic, tt, pp, qq in (("/ground_truth/odom", flight["rtk_t"], flight["rtk_p"], flight["truth_q"][rtk_idx]),
                              ("/lio/odom", flight["lio_t"], flight["lio_p"], flight["lio_q"])):
        for k, t in enumerate(tt):
            h, ns = hdr(t, "map")
            pose = Pose(position=Point(x=float(pp[k, 0]), y=float(pp[k, 1]), z=float(pp[k, 2])), orientation=quat(qq[k]))
            m = Odom(header=h, child_frame_id="base_link", pose=PoseC(pose=pose, covariance=np.zeros(36)),
                     twist=TwistC(twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0)),
                                  covariance=np.zeros(36)))
            msgs.append((topic, Odom, ns, m))
    msgs.sort(key=lambda x: x[2])
    with Writer(out_path, version=8) as writer:
        conns = {}
        for topic, typ, _, _ in msgs:
            if topic not in conns:
                conns[topic] = writer.add_connection(topic, typ.__msgtype__, typestore=ts)
        for topic, typ, ns, m in msgs:
            writer.write(conns[topic], ns, ts.serialize_cdr(m, typ.__msgtype__))
    return out_path
