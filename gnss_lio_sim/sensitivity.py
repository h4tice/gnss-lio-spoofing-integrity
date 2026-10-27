"""Sonradan seçilen füzyon parametresinin duyarlılık analizi.

`fusion.lio_unhealthy_hold_nognss` (GNSS yokken sağlıksız ilan edilen LIO'nun en uzun dışlanma
süresi) gerçek veri sonuçları görüldükten sonra eklendi (2 s). Burada değer {0, 1, 2, 5, ∞}
üzerinde taranır (∞ = eski davranış) ve ana ölçütlerin bu seçime ne kadar bağlı olduğu raporlanır.

    python -m gnss_lio_sim sensitivity
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

VALUES = [0.0, 1.0, 2.0, 5.0, float("inf")]
ATTACKS = ["clean", "jump", "ramp", "slow_drift", "noise", "dropout"]
SPOOF = ["jump", "ramp", "slow_drift", "noise"]
METHODS = ["hard", "soft"]


def _one(args):
    from .fusion import run_fusion
    from .metrics import eval_run
    from .sim.flight import load_flight
    from .spoofing import inject
    cfg_path, path, group, value = args
    cfg = load_config(cfg_path, {"fusion": {"lio_unhealthy_hold_nognss": value}})
    fl = load_flight(path)
    meta = fl["meta"]
    rows = []
    for a in ATTACKS:
        att = inject(fl["gnss_t"], fl["gnss_pos"], fl["gnss_vel"], meta, a, cfg)
        for m in METHODS:
            r = eval_run(fl, att, run_fusion(fl, att, m, cfg), cfg)
            rows.append({"group": group or meta["env"], "id": meta["id"], "hold": value, "attack": a,
                         "method": m, **r})
    return rows


def run(jobs, out_dir, workers, groups=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(c, str(p), g, v) for c, p, g in jobs for v in VALUES]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, tasks, chunksize=1):
            rows += r
    df = pd.DataFrame(rows)
    if groups and (out / "sensitivity_results.csv").exists():  # yalnız bu gruplar yeniden koşturuldu
        old = pd.read_csv(out / "sensitivity_results.csv")
        df = pd.concat([old[~old.group.isin(groups)], df], ignore_index=True)
    df.to_csv(out / "sensitivity_results.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "sensitivity_summary.csv", index=False)
    return df, s


def summarize(df):
    rows = []
    for (g, v), d in df.groupby(["group", "hold"]):
        so, ha = d[d.method == "soft"], d[d.method == "hard"]
        sd = so[so.attack == "slow_drift"]
        sp_s = so[so.attack.isin(SPOOF)].set_index(["id", "attack"])
        sp_h = ha[ha.attack.isin(SPOOF)].set_index(["id", "attack"])
        rows.append({"group": g, "hold_s": v,
                     "far_pct": 100 * so.fa_epochs.sum() / max(so.healthy_epochs.sum(), 1),
                     "jump_delay_s": so[so.attack == "jump"].delay.median(),
                     "slow_before5_pct": 100 * float((sd.detected & (sd.offset_at_det < 5)).mean()),
                     "H2_pct": 100 * (1 - sp_s.jump_post.mean() / sp_h.jump_post.mean()),
                     "peak_add_err_m": sp_s.peak_add_err.mean(),
                     "max_err_m": so[so.attack != "clean"].max_err.max(),
                     "dropout_drift_pct": so[so.attack == "dropout"].drift_pct.mean(),
                     "rmse_clean_m": so[so.attack == "clean"].rmse.mean()})
    return pd.DataFrame(rows)
