"""Yavaş sürükleme hızı taraması: önerilen yöntemin tespit sınırı (en küçük tespit edilebilir hız).

Yöntem ve eşikler değiştirilmez; yalnız `slow_drift` saldırısının hızı taranır. Her uçuş kaydı ve
hız için önerilen yöntem (soft) koşturulur; 60 s'lik saldırı penceresinde tespit olasılığı,
kasıtlı öteleme 5 m'ye ulaşmadan tespit olasılığı ve tespit anındaki öteleme raporlanır.
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .fusion import run_fusion
from .metrics import eval_run
from .sim.flight import load_flight
from .spoofing import inject

RATES = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8]  # [m/s]
LIMIT_M = 5.0
ENV_TR = {"campus": "Benzetim: kampüs", "open_field": "Benzetim: açık arazi",
          "flat_poor": "Benzetim: yapısız düz alan", "real": "Gerçek veri (MARS-LVIG)",
          "semireal": "Yarı-gerçek (AMtown03)", "munfrl": "Gerçek veri (MUN-FRL, tek nokta GNSS)", "munfrl_rtk": "Gerçek veri (MUN-FRL, RTK GNSS)"}
# kategorik sıra sabit (referans paletin ilk dört yuvası); her seri ayrıca işaretçiyle ayrışır
STYLE = {"real": ("#2a78d6", "o"), "campus": ("#eb6834", "s"), "open_field": ("#1baf7a", "^"),
         "flat_poor": ("#eda100", "D"), "semireal": ("#e87ba4", "v"),
         "munfrl": ("#8a5cc2", "P"), "munfrl_rtk": ("#6b6a64", "X")}


def _one(args):
    cfg_path, path, group, rate = args
    cfg = load_config(cfg_path)
    flight = load_flight(path)
    meta = flight["meta"]
    att = inject(flight["gnss_t"], flight["gnss_pos"], flight["gnss_vel"], meta, "slow_drift", cfg,
                 rate / cfg.attacks.slow_drift.rate)
    m = eval_run(flight, att, run_fusion(flight, att, "soft", cfg), cfg)
    return {"group": group, "id": meta["id"], "rate": rate, "detected": m["detected"],
            "offset_at_det": m["offset_at_det"], "delay": m["delay"], "far": m["far"],
            "reach5": rate * meta["denied"] >= LIMIT_M}


def run(jobs, out_dir, workers, groups=None):
    """jobs: [(cfg_path, flight_path, group)]; group 'real' ya da benzetim ortamı (None: meta['env'])."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for cfg_path, path, group in jobs:
        g = group or load_flight(path)["meta"]["env"]
        tasks += [(cfg_path, path, g, r) for r in RATES]
    with ProcessPoolExecutor(workers) as ex:
        df = pd.DataFrame(list(ex.map(_one, tasks, chunksize=2)))
    if groups and (out / "drift_sweep.csv").exists():  # yalnız bu gruplar yeniden koşturuldu
        old = pd.read_csv(out / "drift_sweep.csv")
        df = pd.concat([old[~old.group.isin(groups)], df], ignore_index=True)
    df.to_csv(out / "drift_sweep.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "drift_sweep_summary.csv", index=False)
    plot(summary, out / "drift_sweep.png")
    return df, summary


def summarize(df):
    rows = []
    for (g, r), d in df.groupby(["group", "rate"]):
        early = d.detected & (d.offset_at_det < LIMIT_M)
        rows.append({"group": g, "rate": r, "n": len(d), "p_detect": d.detected.mean(),
                     "p_before5": early.mean() if d.reach5.all() else np.nan,
                     "offset_med": d.offset_at_det.median(), "offset_q25": d.offset_at_det.quantile(0.25),
                     "offset_q75": d.offset_at_det.quantile(0.75), "delay_med": d.delay.median()})
    return pd.DataFrame(rows)


def min_rate(summary, group, level=0.95):
    """5 m öncesi tespit oranının bundan sonra hep ≥ level kaldığı en küçük hız (yoksa NaN)."""
    s = summary[(summary.group == group) & summary.p_before5.notna()].sort_values("rate")
    ok = (s.p_before5 >= level).to_numpy()
    for i in range(len(s)):
        if ok[i:].all():
            return float(s.rate.iloc[i])
    return np.nan


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "font.size": 10})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    order = [g for g in ("real", "munfrl", "munfrl_rtk", "semireal", "campus", "open_field", "flat_poor") if g in set(summary.group)]
    for g in order:
        c, mk = STYLE[g]
        s = summary[summary.group == g].sort_values("rate")
        p = s[s.p_before5.notna()]
        ax[0].plot(p.rate, 100 * p.p_before5, color=c, marker=mk, ms=7, lw=2, label=ENV_TR[g],
                   mec="#fcfcfb", mew=1.5)
        ax[1].plot(s.rate, s.offset_med, color=c, marker=mk, ms=7, lw=2, label=ENV_TR[g], mec="#fcfcfb", mew=1.5)
        ax[1].fill_between(s.rate, s.offset_q25, s.offset_q75, color=c, alpha=0.12, lw=0)
    reach = LIMIT_M / 60.0  # 60 s'lik saldırı penceresinde 5 m'ye ulaşılan en küçük hız
    ax[0].axvspan(RATES[0] * 0.9, reach, color=grid, alpha=0.6, lw=0)
    ax[0].text(np.sqrt(RATES[0] * 0.9 * reach), 50, "60 s'de\n5 m'ye\nulaşılmaz", ha="center", va="center",
               color=muted, fontsize=8.5)
    ax[0].set_xlim(RATES[0] * 0.9, RATES[-1] * 1.1)
    ax[0].axhline(95, color=muted, ls="--", lw=1)
    ax[0].text(0.09, 96.5, "%95", color=muted, fontsize=9)
    ax[0].set_ylim(-3, 103)
    ax[0].set_ylabel("Öteleme 5 m'ye ulaşmadan tespit [%]")
    ax[0].set_title("Tespit olasılığı", color=ink, loc="left")
    ax[1].axhline(LIMIT_M, color=muted, ls="--", lw=1)
    ax[1].text(0.052, LIMIT_M + 0.25, "5 m sınırı", color=muted, fontsize=9)
    ax[1].set_ylabel("Tespit anındaki kasıtlı öteleme [m] (medyan, IQR)")
    ax[1].set_title("Tespit anında öteleme", color=ink, loc="left")
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(RATES)
        a.set_xticklabels([f"{r:g}" for r in RATES])
        a.minorticks_off()
        a.set_xlabel("Sürükleme hızı [m/s] (log)")
        a.grid(color=grid, lw=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
