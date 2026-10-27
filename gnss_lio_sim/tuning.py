"""Tespit parametrelerinin ayarı (öneri İP2 B planı: eşik/pencere parametrelerinin ROC
tabanlı yeniden ayarlanması).

Geliştirme kümesi (hızlı profil uçuşları) üzerinde tespit parametre ızgarası taranır;
yanlış alarm oranı kısıtı altında yavaş sürüklemeli aldatmayı 5 m öteleme aşılmadan
tespit oranını en yükselten kombinasyon seçilir. Tam deneyin uçuşları farklı tohumlardan
üretildiği için seçim test verisine sızmaz.
"""
import itertools
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from .config import load_config
from .fusion import run_fusion
from .metrics import eval_run
from .sim.flight import load_flight
from .spoofing import inject

GRID = {
    "disp_max_deg_frac": [0.5, 0.75],
    "disp_alpha": [1e-3, 1e-4],
    "vel_alpha": [1e-3, 1e-5],
    "soft_onset_chi2": [0.7, 0.85],
    "confirm_epochs": [5],
    "cusum": [(1.5, 25.0)],
}


def _combos():
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        yield dict(zip(keys, vals))


def _apply(cfg, combo):
    for k, v in combo.items():
        if k == "cusum":
            cfg.detector.cusum_k, cfg.detector.cusum_h = v
        elif k.startswith("soft_"):
            cfg.transition[k] = v
        else:
            cfg.detector[k] = v


def _job(args):
    cfg_path, overrides, path, ci, combo, method = args
    cfg = load_config(cfg_path, overrides)
    _apply(cfg, combo)
    fl = load_flight(path)
    out = []
    for attack in ("clean", "slow_drift", "jump"):
        att = inject(fl["gnss_t"], fl["gnss_pos"], fl["gnss_vel"], fl["meta"], attack, cfg)
        m = eval_run(fl, att, run_fusion(fl, att, method, cfg), cfg)
        out.append({"combo": ci, "id": fl["meta"]["id"], "env": fl["meta"]["env"], "attack": attack, **m})
    return out


def tune(cfg_path, overrides, flights_dir, out_dir, workers, method="soft", far_max=0.01, log=print):
    flights = sorted(Path(flights_dir).glob("*.npz"))
    combos = list(_combos())
    jobs = [(cfg_path, overrides, str(p), ci, c, method) for ci, c in enumerate(combos) for p in flights]
    log(f"[tune] {len(combos)} parametre kombinasyonu × {len(flights)} uçuş ({method} yöntemi)")
    rows = []
    with get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_job, jobs, chunksize=4):
            rows += r
    df = pd.DataFrame(rows)
    summ = []
    for ci, g in df.groupby("combo"):
        sd = g[g.attack == "slow_drift"]
        jd = g[g.attack == "jump"]
        summ.append({**{k: str(v) for k, v in combos[ci].items()},
                     "far": g.fa_epochs.sum() / max(g.healthy_epochs.sum(), 1),
                     "far_flat": (lambda x: x.fa_epochs.sum() / max(x.healthy_epochs.sum(), 1))(g[g.env == "flat_poor"]),
                     "slow_before_5m": float(((sd.offset_at_det < 5) & sd.detected).mean()),
                     "slow_delay": sd.delay.mean(), "jump_delay": jd.delay.mean(), "combo": ci})
    s = pd.DataFrame(summ)
    s.to_csv(Path(out_dir) / "tuning.csv", index=False)
    ok = s[s.far <= far_max]
    best = (ok if len(ok) else s).sort_values(["slow_before_5m", "far"], ascending=[False, True]).iloc[0]
    log(s.drop(columns="combo").round(4).to_string(index=False))
    log(f"[tune] seçilen ({'YA kısıtı sağlandı' if len(ok) else 'YA kısıtı sağlanamadı'}): {combos[int(best.combo)]} "
        f"(YA={100 * best.far:.2f}%, 5 m öncesi tespit={100 * best.slow_before_5m:.0f}%)")
    return combos[int(best.combo)]
