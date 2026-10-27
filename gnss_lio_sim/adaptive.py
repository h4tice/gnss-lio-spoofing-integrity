"""Uyarlanabilir (beyaz kutu) saldırgan testi: dedektörün eşiğinin hemen altında kalarak sürükleyen aldatıcı.

Saldırgan hedef yöntemin normalize karar istatistiğini (ρ) her çağda gözler ve ötelemeyi ρ'yu `margin`
altında tutacak hızla büyütür (bkz. ``spoofing.AdaptiveAttacker``). Saldırgan ayarı (margin, r_max) bir
ızgarada taranır; her uçuş ve yöntem için saldırganın **en iyi** ayarı (en büyük ek seyrüsefer hatası)
raporlanır (en kötü durum). Ölçütler:

    detected        : saldırı penceresinde alarm
    undetected_off  : alarmdan önce (ya da hiç alarm yoksa pencere sonunda) ulaşılan yatay öteleme [m]
    peak_add_err    : saldırı penceresinde seyrüsefer hatasındaki en büyük artış [m]

    python -m gnss_lio_sim adaptive [--groups real munfrl ...]
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

MARGINS = [0.3, 0.5, 0.8]
RATES = [0.2, 1.0]  # saldırganın en büyük öteleme hızı [m/s]
# (etiket, run_fusion yöntemi, yapılandırma üzerine yazma)
METHODS = [("gnss_ins", "gnss_ins", None), ("ci_ins", "ci_ins", None),
           ("gu_lio_b30", "gu_lio", {"baselines": {"gu": {"beta": 30.0}}}),
           ("alja_lio", "alja_lio", None), ("hard", "hard", None), ("soft", "soft", None)]
LABELS = {"gnss_ins": "GNSS/INS (dışlamasız)", "ci_ins": "Tanil ve ark. 2018", "gu_lio_b30": "Gu ve ark. 2021 (β=30)",
          "alja_lio": "Aljaafreh ve ark. 2026", "hard": "Sert anahtarlama", "soft": "Önerilen yöntem"}


def _one(args):
    from .fusion import run_fusion
    from .metrics import eval_run
    from .sim.flight import load_flight
    from .spoofing import inject_adaptive
    cfg_path, path, group = args
    fl = load_flight(path)
    meta = fl["meta"]
    rows = []
    for label, method, ov in METHODS:
        cfg = load_config(cfg_path, ov)
        for margin in MARGINS:
            for r_max in RATES:
                att = inject_adaptive(fl["gnss_t"], fl["gnss_pos"], fl["gnss_vel"], meta, cfg, margin, r_max)
                m = eval_run(fl, att, run_fusion(fl, att, method, cfg), cfg)
                t_on, t_off = att["window"]
                w = (att["t"] >= t_on) & (att["t"] < t_off)
                off = np.linalg.norm(att["offset"][:, :2], axis=1)
                t_end = m["t_alarm"] if m["detected"] else t_off
                u = w & (att["t"] < t_end)
                rows.append({"group": group or meta["env"], "id": meta["id"], "method": label, "margin": margin,
                             "r_max": r_max, "detected": m["detected"], "delay": m["delay"],
                             "undetected_off": float(off[u].max()) if u.any() else 0.0,
                             "off_end": float(off[w][-1]) if w.any() else 0.0,
                             "peak_add_err": m["peak_add_err_attack"],
                             "peak_add_err_all": m["peak_add_err"]})
    return rows


def run(jobs, out_dir, workers, groups=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, jobs, chunksize=1):
            rows += r
    df = pd.DataFrame(rows)
    if groups and (out / "adaptive_results.csv").exists():  # yalnız bu gruplar yeniden koşturuldu
        old = pd.read_csv(out / "adaptive_results.csv")
        df = pd.concat([old[~old.group.isin(groups)], df], ignore_index=True)
    df.to_csv(out / "adaptive_results.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "adaptive_summary.csv", index=False)
    plot(s, out / "adaptive.png")
    return df, s


def summarize(df):
    """Her uçuş × yöntem için saldırganın en iyi ayarı (en büyük ek hata); sonra uçuşlar üzerinden özet."""
    worst = df.loc[df.groupby(["group", "id", "method"]).peak_add_err.idxmax()]
    rows = []
    for (g, m), d in worst.groupby(["group", "method"]):
        a = df[(df.group == g) & (df.method == m)]
        rows.append({"group": g, "method": m, "n_flights": len(d),
                     "worst_add_err_mean": d.peak_add_err.mean(), "worst_add_err_max": d.peak_add_err.max(),
                     "worst_detected_pct": 100 * d.detected.mean(),
                     "undetected_off_mean": d.undetected_off.mean(),
                     "det_pct_all_settings": 100 * a.detected.mean()})
    return pd.DataFrame(rows)


GROUPS = {"real": "Gerçek veri (MARS-LVIG)", "munfrl": "Gerçek veri (MUN-FRL, tek nokta GNSS)",
          "munfrl_rtk": "Gerçek veri (MUN-FRL, RTK GNSS)", "semireal": "Yarı-gerçek (AMtown03)",
          "campus": "Benzetim: kampüs", "open_field": "Benzetim: açık arazi", "flat_poor": "Benzetim: yapısız düz alan"}


def plot(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "font.size": 9})
    groups = [g for g in GROUPS if g in set(s.group)]
    order = [m for m, _, _ in METHODS]
    fig, ax = plt.subplots(1, len(groups), figsize=(2.9 * len(groups), 3.6), squeeze=False, sharey=True)
    for j, g in enumerate(groups):
        a = ax[0, j]
        d = s[s.group == g].set_index("method").reindex(order)
        y = np.arange(len(order))
        col = ["#2a78d6" if m == "soft" else "#b9b8b1" for m in order]
        a.barh(y, d.worst_add_err_mean, color=col, height=0.6)
        for yi, (v, det) in enumerate(zip(d.worst_add_err_mean, d.worst_detected_pct)):
            if np.isfinite(v):
                a.text(v * 1.15, yi, f"{v:.1f} m · %{det:.0f}", va="center", fontsize=7, color=ink)
        a.set_xscale("log")
        a.set_xlim(0.1, 400)
        a.set_yticks(y)
        a.set_yticklabels([LABELS[m] for m in order])
        a.invert_yaxis()
        a.grid(axis="x", color=grid, lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.set_title(GROUPS[g], color=ink, loc="left", fontsize=8.5)
        a.set_xlabel("Ek hata [m] (log)")
    fig.suptitle("Uyarlanabilir saldırgan, en kötü durum: saldırganın en iyi ayarında ek seyrüsefer hatası "
                 "(etiket: o ayarda tespit oranı)", x=0.01, ha="left", fontsize=9.5, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
