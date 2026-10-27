"""Tek bir uçuşun benzetimi: ortam + yörünge + algılayıcılar + çevrim içi LIO.

Uçuş kaydı (npz) sağlıklı GNSS ölçümlerini içerir; aldatma senaryoları daha sonra
kayıt-üzerinde-oynatma (record & replay) ile enjekte edilir — öneride tanımlanan
deney yöntemiyle aynı.
"""
import json
import time
import zlib
from pathlib import Path

import numpy as np

from ..config import to_plain
from ..geometry import rot_to_quat
from ..lio import LIO
from .environment import build_environment
from .sensors import Lidar, simulate_gnss, simulate_imu, simulate_rtk
from .trajectory import Trajectory


def flight_id(env, traj, speed, rep):
    return f"{env}_{traj}_v{speed:g}_r{rep}"


def flight_seed(base, env, traj, speed, rep):
    return (zlib.crc32(flight_id(env, traj, speed, rep).encode()) + base) % (2 ** 31)


def build_world(cfg, env_type, traj_name, speed, rep):
    """Tohumdan belirlenimci olarak: rng, saldırı başlangıcı/yönü, yörünge ve ortam."""
    seed = flight_seed(cfg.seed, env_type, traj_name, speed, rep)
    rng = np.random.default_rng(seed)
    fc = cfg.flight
    onset = float(rng.uniform(*fc.attack_onset))
    duration = fc.attack_onset[1] + fc.gnss_denied_duration + fc.recovery_duration
    attack_dir = rng.normal(size=2)
    attack_dir = np.r_[attack_dir / np.linalg.norm(attack_dir), 0.0]
    traj = Trajectory(traj_name, speed, fc.altitude, duration, rng, jitter=fc.path_jitter)
    env = build_environment(env_type, cfg.environment.size, traj.path_xy(), rng)
    return {"seed": seed, "rng": rng, "onset": onset, "duration": duration,
            "attack_dir": attack_dir, "traj": traj, "env": env}


def simulate_flight(cfg, env_type, traj_name, speed, rep, out_dir, save_scans=False,
                    progress=None):
    fid = flight_id(env_type, traj_name, speed, rep)
    w = build_world(cfg, env_type, traj_name, speed, rep)
    seed, rng, onset, duration = w["seed"], w["rng"], w["onset"], w["duration"]
    attack_dir, traj, env = w["attack_dir"], w["traj"], w["env"]
    fc = cfg.flight
    truth = traj.generate(rate=1000.0)
    imu = simulate_imu(truth, cfg.imu, rng)
    gnss = simulate_gnss(truth, cfg.gnss, rng)
    rtk = simulate_rtk(truth, cfg.rtk.sigma, cfg.lidar.rate, rng)
    lidar = Lidar(cfg.lidar, env)

    imu_step = int(round(truth["rate"] / cfg.imu.rate))
    scan_every = int(round(cfg.imu.rate / cfg.lidar.rate))
    ti = imu["t"]
    lio = LIO(cfg.lio, cfg.imu, truth["p"][0], truth["R"][0], truth["v"][0])
    recs, scans = [], []
    tic = time.perf_counter()
    for k in range(1, len(ti)):
        lio.propagate(imu["acc"][k - 1], imu["gyro"][k - 1], ti[k] - ti[k - 1])
        if k % scan_every == 0:
            j = k * imu_step
            scan = lidar.scan(truth["p"][j], truth["R"][j], rng)
            recs.append(lio.update(ti[k], scan))
            if save_scans:
                scans.append(scan)
            if progress is not None and len(recs) % 100 == 0:
                progress(fid, ti[k] / duration)
    wall = time.perf_counter() - tic

    # ---- kayıt
    idx = np.arange(0, len(truth["t"]), imu_step)
    lio_arr = {f"lio_{key}": np.array([r[key] for r in recs])
               for key in recs[0] if key not in ("R",)}
    lio_arr["lio_q"] = rot_to_quat(np.array([r["R"] for r in recs]))
    meta = {
        "id": fid, "env": env_type, "traj": traj_name, "speed": speed, "rep": rep,
        "seed": seed, "onset": onset, "duration": duration,
        "denied": fc.gnss_denied_duration, "attack_dir": attack_dir.tolist(),
        "path_length": traj.length, "env_meta": env.meta,
        "sim_wall_time": wall, "config": to_plain(cfg),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{fid}.npz",
        truth_t=truth["t"][idx], truth_p=truth["p"][idx], truth_v=truth["v"][idx],
        truth_q=rot_to_quat(truth["R"][idx]),
        imu_t=imu["t"], imu_acc=imu["acc"], imu_gyro=imu["gyro"], imu_ba=imu["ba"], imu_bg=imu["bg"],
        gnss_t=gnss["t"], gnss_pos=gnss["pos"], gnss_vel=gnss["vel"],
        rtk_t=rtk["t"], rtk_p=rtk["p"], **lio_arr,
        meta=json.dumps(meta),
    )
    if save_scans:
        sd = out_dir / "scans"
        sd.mkdir(exist_ok=True)
        lens = np.array([len(s) for s in scans])
        np.savez_compressed(sd / f"{fid}_scans.npz", t=np.array([r["t"] for r in recs]),
                            lengths=lens, points=np.concatenate(scans).astype(np.float32))
    return fid, wall


def load_flight(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def load_scans(path):
    d = np.load(path)
    splits = np.cumsum(d["lengths"])[:-1]
    return d["t"], np.split(d["points"], splits)
