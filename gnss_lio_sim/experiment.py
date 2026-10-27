"""Deney koşturucusu.

1) simulate : ortam × yörünge × hız × tekrar uçuşlarını üretir (LIO çevrim içi çalışır)
2) evaluate : her uçuş kaydı üzerinde, aldatma senaryoları × füzyon yöntemleri
              kayıt-üzerinde-oynatma ile koşturulur; metrikler ve zaman serileri kaydedilir
"""
import itertools
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .fusion import METHODS, run_fusion
from .metrics import _plane, drift_segments, eval_lio_only, eval_run
from .sim.flight import flight_id, load_flight, simulate_flight
from .spoofing import ATTACK_TYPES, inject

SEVERITY_ATTACKS = ["jump", "ramp", "slow_drift"]
SEVERITY_METHODS = ["hard", "soft"]


def flight_plan(cfg):
    fc = cfg.flight
    return [(e, tr, float(s), r) for e, tr, s, r in itertools.product(
        cfg.environment.types, fc.trajectories, fc.speeds, range(fc.repetitions))]


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


# ------------------------------------------------------------------ benzetim
def _sim_job(args):
    cfg_path, overrides, plan_item, out_dir, save_scans = args
    cfg = load_config(cfg_path, overrides)
    fid, wall = simulate_flight(cfg, *plan_item, out_dir, save_scans=save_scans)
    return fid, wall


def simulate_all(cfg, cfg_path, overrides, out_dir, workers=None, save_scans=False, log=print):
    out_dir = Path(out_dir)
    plan = flight_plan(cfg)
    todo = [p for p in plan if not (out_dir / f"{flight_id(*p)}.npz").exists()]
    log(f"[simulate] {len(plan)} uçuş planlandı, {len(todo)} tanesi üretilecek "
        f"({workers or default_workers()} işçi)")
    if not todo:
        return
    jobs = [(cfg_path, overrides, p, str(out_dir), save_scans) for p in todo]
    tic = time.time()
    with get_context("spawn").Pool(workers or default_workers()) as pool:
        for i, (fid, wall) in enumerate(pool.imap_unordered(_sim_job, jobs), 1):
            log(f"  [{i}/{len(todo)}] {fid}  ({wall:.1f} s)  toplam {time.time() - tic:.0f} s")


# ------------------------------------------------------------------ değerlendirme
def evaluate_flight(cfg, path, runs_dir, severity=True):
    flight = load_flight(path)
    meta = flight["meta"]
    base = {k: meta[k] for k in ("id", "env", "traj", "speed", "rep", "onset")}
    rows = []
    series = {}

    lio = eval_lio_only(flight, cfg)
    rows.append({**base, "attack": "clean", "severity": 1.0, "method": "lio_only", **lio})
    ref_t = flight["lio_t"]
    series["lio_only/err"] = np.linalg.norm(flight["lio_p"] - flight["rtk_p"][
        np.clip(np.searchsorted(flight["rtk_t"], ref_t - 1e-6), 0, len(flight["rtk_t"]) - 1)], axis=1)

    scenarios = [("clean", 1.0, m) for m in METHODS]
    scenarios += [(a, 1.0, m) for a in ATTACK_TYPES for m in METHODS]
    if severity:
        for a in SEVERITY_ATTACKS:
            for s in cfg.attacks.severity_scales:
                if s != 1.0:
                    scenarios += [(a, float(s), m) for m in SEVERITY_METHODS]
    for attack, sev, method in scenarios:
        att = inject(flight["gnss_t"], flight["gnss_pos"], flight["gnss_vel"], meta, attack, cfg, sev)
        res = run_fusion(flight, att, method, cfg)
        m = eval_run(flight, att, res, cfg)
        rows.append({**base, "attack": attack, "severity": sev, "method": method, **m})
        if sev == 1.0:
            key = f"{attack}/{method}"
            ref = flight["rtk_p"][np.clip(np.searchsorted(flight["rtk_t"], res["t"] - 1e-6), 0,
                                          len(flight["rtk_t"]) - 1)]
            series[f"{key}/err"] = np.linalg.norm(_plane(cfg, res["p"] - ref), axis=1).astype(np.float32)
            series[f"{key}/p"] = res["p"].astype(np.float32)
            for g in ("rA", "rB", "rC", "rD", "rho", "lam", "alarm"):
                series[f"{key}/{g}"] = res[f"g_{g}"].astype(np.float32)
            series[f"{key}/gnss_pos"] = att["pos"].astype(np.float32)
            series[f"{key}/gnss_valid"] = att["valid"]
    series["t"] = res["t"]
    series["g_t"] = res["g_t"]
    series["ref_p"] = flight["rtk_p"][np.clip(np.searchsorted(flight["rtk_t"], res["t"] - 1e-6), 0,
                                              len(flight["rtk_t"]) - 1)].astype(np.float32)
    series["lio_p"] = flight["lio_p"].astype(np.float32)
    series["lio_t"] = flight["lio_t"]
    segs = [{**base, **s} for s in drift_segments(flight, cfg)]
    np.savez_compressed(Path(runs_dir) / f"{meta['id']}.npz", meta=json.dumps(
        {k: meta[k] for k in ("id", "env", "traj", "speed", "rep", "onset", "denied", "attack_dir")}),
        **series)
    return rows, segs


def _eval_job(args):
    cfg_path, overrides, path, runs_dir, severity = args
    cfg = load_config(cfg_path, overrides)
    return evaluate_flight(cfg, path, runs_dir, severity)


def evaluate_all(cfg, cfg_path, overrides, flights_dir, out_dir, workers=None, severity=True,
                 log=print):
    flights = sorted(Path(flights_dir).glob("*.npz"))
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log(f"[evaluate] {len(flights)} uçuş kaydı üzerinde senaryolar koşturuluyor")
    jobs = [(cfg_path, overrides, str(p), str(runs_dir), severity) for p in flights]
    rows, segs = [], []
    tic = time.time()
    with get_context("spawn").Pool(workers or default_workers()) as pool:
        for i, (r, s) in enumerate(pool.imap_unordered(_eval_job, jobs), 1):
            rows += r
            segs += s
            log(f"  [{i}/{len(jobs)}] {r[0]['id']}  toplam {time.time() - tic:.0f} s")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(segs).to_csv(out_dir / "drift_segments.csv", index=False)
    log(f"[evaluate] {len(df)} koşu -> {out_dir / 'results.csv'}")
    return df
