"""Yalnız-tespit ROC/AUC karşılaştırması (Aljaafreh ve ark. 2026'daki "detection-only replay" protokolünün eşi).

Her uçuş kaydında saldırısız ve saldırılı senaryolar, GNSS hiç dışlanmadan (``detect_only``) koşturulur;
her tespit yönteminin karar istatistiği aynı süzgeç durumu üzerinde hesaplanır. Bir örneğin puanı, saldırı
penceresi [başlangıç, başlangıç + süre] boyunca karar istatistiğinin en büyük değeridir (saldırısız
örnekte aynı pencere). Karar istatistikleri eşik = 1 olacak biçimde normalize edilir ve yöntemin kendi
süreklilik kuralını içerir:

    soft     : önerilen yöntem; ρ = max(rA, rB, rC, rD), confirm_epochs ardışık çağda en küçük değer
    ci_ins   : Tanil 2018; q / χ² eşiği, confirm_epochs ardışık çağda en küçük değer (aynı karar kuralı)
    gu_lio   : Gu 2021; l / (β·Th) (2 s'de bir güncellenir, süreklilik kuralı yok)
    alja_lio : Aljaafreh 2026; max(N_G ardışık çağda en küçük J/γ, g/h)

    python -m gnss_lio_sim roc
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .config import load_config

SPOOF = ["jump", "ramp", "slow_drift", "noise"]
# (etiket, run_fusion yöntemi)
DETECTORS = [("soft", "monitor"), ("ci_ins", "ci_ins"), ("gu_lio", "gu_lio"), ("alja_lio", "alja_lio")]
LABELS = {"soft": "Önerilen yöntem", "ci_ins": "Tanil ve ark. 2018", "gu_lio": "Gu ve ark. 2021",
          "alja_lio": "Aljaafreh ve ark. 2026"}


def _runmin(x, n):
    """n ardışık örnekte en küçük değer (süreklilik kuralı: n çağ boyunca eşiği aşmış olmak)."""
    if n <= 1 or len(x) < n:
        return x
    w = np.lib.stride_tricks.sliding_window_view(x, n).min(axis=1)
    return np.r_[np.full(n - 1, -np.inf), w]


def score_series(label, res, cfg):
    if label == "alja_lio":
        return np.maximum(_runmin(res["g_rA"], cfg.baselines.alja.n_g), res["g_rB"])
    if label == "gu_lio":
        return res["g_rho"]
    return _runmin(res["g_rho"], cfg.detector.confirm_epochs)


def _one(args):
    from .fusion import run_fusion
    from .sim.flight import load_flight
    from .spoofing import inject
    cfg_path, path, group = args
    cfg = load_config(cfg_path)
    fl = load_flight(path)
    meta = fl["meta"]
    t_on, t_off = meta["onset"], meta["onset"] + meta["denied"]
    rows = []
    for a in ["clean"] + SPOOF:
        att = inject(fl["gnss_t"], fl["gnss_pos"], fl["gnss_vel"], meta, a, cfg)
        for label, method in DETECTORS:
            res = run_fusion(fl, att, method, cfg, detect_only=True)
            s = score_series(label, res, cfg)
            w = (res["g_t"] >= t_on) & (res["g_t"] < t_off) & np.isfinite(s)
            rows.append({"group": group or meta["env"], "id": meta["id"], "attack": a, "detector": label,
                         "score": float(s[w].max()) if w.any() else np.nan})
    return rows


def run(jobs, out_dir, workers, groups=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, jobs, chunksize=1):
            rows += r
    df = pd.DataFrame(rows)
    if groups and (out / "roc_scores.csv").exists():  # yalnız bu gruplar yeniden koşturuldu
        old = pd.read_csv(out / "roc_scores.csv")
        df = pd.concat([old[~old.group.isin(groups)], df], ignore_index=True)
    df.to_csv(out / "roc_scores.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "roc_summary.csv", index=False)
    plot(df, out / "roc.png")
    return df, s


def _auc(d, attacks):
    neg = d[d.attack == "clean"].score.to_numpy()
    pos = d[d.attack.isin(attacks)].score.to_numpy()
    neg, pos = neg[np.isfinite(neg)], pos[np.isfinite(pos)]
    if not len(neg) or not len(pos):
        return np.nan, np.nan, np.nan
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    x = np.r_[neg, pos]
    # eşik = 1 (yöntemin kendi işletim noktası): tespit oranı ve örnek bazında yanlış alarm
    return roc_auc_score(y, x), float((pos >= 1).mean()), float((neg >= 1).mean())


def summarize(df):
    rows = []
    for (g, det), d in df.groupby(["group", "detector"]):
        r = {"group": g, "detector": det, "n_clean": int((d.attack == "clean").sum())}
        r["auc_all"], r["pd_all"], r["pfa_run"] = _auc(d, SPOOF)
        for a in SPOOF:
            r[f"auc_{a}"], r[f"pd_{a}"], _ = _auc(d, [a])
        rows.append(r)
    return pd.DataFrame(rows)


GROUPS = {"real": "Gerçek veri (MARS-LVIG)", "munfrl": "Gerçek veri (MUN-FRL, tek nokta GNSS)", "munfrl_rtk": "Gerçek veri (MUN-FRL, RTK GNSS)", "semireal": "Yarı-gerçek (AMtown03)", "campus": "Benzetim: kampüs",
          "open_field": "Benzetim: açık arazi", "flat_poor": "Benzetim: yapısız düz alan"}
# kategorik sıra sabit (referans paletin ilk dört yuvası); önerilen yöntem birinci yuvada
STYLE = {"soft": ("#2a78d6", "-"), "ci_ins": ("#eb6834", "--"), "gu_lio": ("#1baf7a", "-."),
         "alja_lio": ("#eda100", ":")}


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "font.size": 9})
    groups = [g for g in GROUPS if g in set(df.group)]
    cols = [("Tüm saldırılar", SPOOF), ("Yavaş sürükleme", ["slow_drift"])]
    fig, ax = plt.subplots(len(cols), len(groups), figsize=(3.0 * len(groups), 6.0), squeeze=False)
    for j, g in enumerate(groups):
        for i, (title, attacks) in enumerate(cols):
            a = ax[i, j]
            for det, _ in DETECTORS:
                d = df[(df.group == g) & (df.detector == det)]
                neg = d[d.attack == "clean"].score.dropna().to_numpy()
                pos = d[d.attack.isin(attacks)].score.dropna().to_numpy()
                if not len(neg) or not len(pos):
                    continue
                fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])
                auc = roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])
                c, ls = STYLE[det]
                a.plot(fpr, tpr, color=c, ls=ls, lw=2, label=f"{LABELS[det]} ({auc:.3f})")
            a.plot([0, 1], [0, 1], color=grid, lw=1)
            a.set_xlim(-0.02, 1.02)
            a.set_ylim(-0.02, 1.02)
            a.grid(color=grid, lw=0.6)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.set_title(f"{GROUPS[g]}\n{title}", color=ink, loc="left", fontsize=8.5)
            a.legend(fontsize=6.5, frameon=False, loc="lower right", title="AUC", title_fontsize=6.5)
            if i == len(cols) - 1:
                a.set_xlabel("Yanlış alarm oranı (örnek)")
            if j == 0:
                a.set_ylabel("Tespit oranı")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
