"""English figures for the GPS Solutions manuscript, drawn directly from the result files.

    python paper/make_figures.py          (from the project folder, inside the venv)

Writes paper/figures/Fig<N>.png (300 dpi, for the .docx) and Fig<N>.eps (vector, for submission).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
RES = ROOT / "results"

INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"
# categorical slots in fixed order; the proposed method always takes the first slot
BLUE, ORANGE, GREEN, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f8e88"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 8.5, "axes.titlecolor": INK, "legend.frameon": False})

DET = [("soft", "Proposed", BLUE, "-"), ("gu_lio", "Gu et al. (2021)", GREEN, "-."),
       ("alja_lio", "Alja'afreh et al. (2026)", YELLOW, ":"), ("ci_ins", "Tanil et al. (2018)", ORANGE, "--")]
GROUP_EN = {"real": "MARS-LVIG (standalone)", "munfrl": "MUN-FRL (single-point)",
            "munfrl_rtk": "MUN-FRL (RTK-fixed)", "semireal": "MARS-LVIG AMtown03 (semi-real)",
            "campus": "Simulation: campus", "open_field": "Simulation: open field",
            "flat_poor": "Simulation: flat, featureless"}


def _clean(a):
    a.grid(color=GRID, lw=0.6)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)


def _save(fig, n):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"Fig{n}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"Fig{n}.eps", facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1: architecture
def fig_architecture():
    fig, a = plt.subplots(figsize=(7.0, 2.9))
    a.set_xlim(0, 100)
    a.set_ylim(0, 42)
    a.axis("off")

    def box(x, y, w, h, text, fc="#f3f6fa", ec=MUTED, bold=False):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color=INK,
               fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x0, y0, x1, y1, label=None, dy=1.2):
        a.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=MUTED))
        if label:
            a.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=6.3, color=MUTED)

    box(1, 31, 14, 8, "IMU\n200-400 Hz")
    box(1, 17, 14, 8, "LiDAR\n10 Hz")
    box(1, 3, 14, 8, "GNSS PVT\n5 Hz")
    box(24, 17, 17, 8, "LiDAR-inertial\nodometry (IEKF)")
    box(50, 24, 22, 15, "Error-state KF\n(15 + 6 states,\nstochastic cloning)", bold=True)
    box(47, 2, 27, 17, "Integrity monitor\nA  windowed $\\chi^2$ (pos.)\nB  CUSUM (pos.+vel.)\n"
        "C  windowed $\\chi^2$ (vel.)\nD  LiDAR-GNSS displacement", fc="#eef5ff", ec=BLUE)
    box(81, 3.5, 18, 14, "Continuous\nde-weighting\n$R_k \\leftarrow \\lambda_k R_k$", fc="#eef5ff", ec=BLUE, bold=True)
    box(81, 27, 18, 9, "Navigation\nsolution")
    arrow(15, 35, 50, 33, "propagation")
    arrow(15, 21, 24, 21)
    arrow(41, 22.5, 50, 28)
    a.text(35, 27.5, "relative pose", fontsize=6.3, color=MUTED)
    arrow(41, 19.5, 47, 13)
    a.text(27, 12.5, "displacement $\\Delta p^{L}$", fontsize=6.3, color=MUTED)
    arrow(15, 7, 47, 7, "position, velocity")
    arrow(61, 24, 61, 19.4)
    a.text(62, 21, "innovations", fontsize=6.3, color=MUTED)
    arrow(74, 10, 81, 10, "$\\rho_k$")
    arrow(90, 18, 72.5, 30, "$\\lambda_k$", dy=0.5)
    arrow(72, 32, 81, 32)
    _save(fig, 1)


# ------------------------------------------------------------------ Fig 2: LIO-only trajectories
def fig_lio():
    sets = [(RES / "real" / "HKairport_GNSS01_lio.npz", "MARS-LVIG HKairport_GNSS01\n(Livox Avia, DJI M300)"),
            (RES / "munfrl" / "lighthouse_lio.npz", "MUN-FRL lighthouse\n(Velodyne VLP-16, DJI M600)")]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for a, (f, title) in zip(ax, sets):
        z = np.load(f)
        gt, p = z["s_gt"], z["s_p_lio"]
        o = gt[0]
        a.plot(gt[:, 0] - o[0], gt[:, 1] - o[1], color=GRAY, lw=2.2, label="Reference (RTK / PPK)")
        a.plot(p[:, 0] - o[0], p[:, 1] - o[1], color=BLUE, lw=1.0, label="LiDAR-inertial odometry (no GNSS)")
        a.set_aspect("equal", adjustable="datalim")
        a.set_xlabel("East [m]")
        a.set_ylabel("North [m]")
        a.set_title(title, loc="left")
        _clean(a)
    ax[0].legend(loc="lower left", fontsize=6.8)
    fig.tight_layout()
    _save(fig, 2)


# ------------------------------------------------------------------ Fig 3: example time series (real flight)
def fig_timeseries(flight="HKisland_GNSS02_o0"):
    from gnss_lio_sim.config import load_config
    from gnss_lio_sim.fusion import run_fusion
    from gnss_lio_sim.metrics import reference_at
    from gnss_lio_sim.sim.flight import load_flight
    from gnss_lio_sim.spoofing import inject
    cfg = load_config(ROOT / "config" / "mars_lvig.yaml")
    fl = load_flight(RES / "real" / "flights" / f"{flight}.npz")
    att = inject(fl["gnss_t"], fl["gnss_pos"], fl["gnss_vel"], fl["meta"], "slow_drift", cfg)
    t_on, t_off = att["window"]
    runs = {m: run_fusion(fl, att, m, cfg) for m in ("gnss_ins", "hard", "soft")}
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 5.4), sharex=True)
    lo, hi = t_on - 20, t_off + 25
    gi = (att["t"] >= lo) & (att["t"] <= hi)
    ax[0].plot(att["t"][gi] - t_on, np.linalg.norm(att["offset"][gi, :2], axis=1), color=GRAY, lw=1.2, ls="--",
               label="Injected offset")
    for m, lab, c in (("gnss_ins", "GNSS/INS", ORANGE), ("hard", "Hard switching", GREEN), ("soft", "Proposed", BLUE)):
        r = runs[m]
        k = (r["t"] >= lo) & (r["t"] <= hi)
        e = np.linalg.norm((r["p"][k] - reference_at(fl, r["t"][k]))[:, :2], axis=1)
        ax[0].plot(r["t"][k] - t_on, e, color=c, lw=1.2, label=lab)
    ax[0].set_ylabel("Horizontal error [m]")
    ax[0].legend(ncol=4, fontsize=6.8, loc="upper left")
    r = runs["soft"]
    k = (r["g_t"] >= lo) & (r["g_t"] <= hi)
    for key, lab, c, ls in (("g_rA", "A", ORANGE, "--"), ("g_rB", "B", YELLOW, ":"), ("g_rC", "C", GREEN, "-."),
                            ("g_rD", "D", BLUE, "-")):
        ax[1].plot(r["g_t"][k] - t_on, r[key][k], color=c, ls=ls, lw=1.1, label=f"Test {lab}")
    ax[1].axhline(1.0, color=INK, lw=0.7)
    ax[1].set_ylim(0, 3)
    ax[1].set_ylabel("Normalized statistic $r$")
    ax[1].legend(ncol=4, fontsize=6.8, loc="upper left")
    ax[2].semilogy(r["g_t"][k] - t_on, r["g_lam"][k], color=BLUE, lw=1.2)
    ax[2].set_ylabel("Inflation $\\lambda$ (proposed)")
    ax[2].set_xlabel("Time since attack onset [s]")
    for a in ax:
        a.axvspan(0, t_off - t_on, color="#fbe9e0", lw=0, zorder=0)
        _clean(a)
    fig.tight_layout()
    _save(fig, 3)


# ------------------------------------------------------------------ Fig 4: ROC, slow drift
def fig_roc(groups=("real", "munfrl", "munfrl_rtk", "flat_poor")):
    df = pd.read_csv(RES / "roc" / "roc_scores.csv")
    fig, ax = plt.subplots(1, len(groups), figsize=(7.2, 2.35), sharey=True)
    for a, g in zip(ax, groups):
        aucs = []
        for det, lab, c, ls in DET:
            d = df[(df.group == g) & (df.detector == det)]
            neg = d[d.attack == "clean"].score.dropna().to_numpy()
            pos = d[d.attack == "slow_drift"].score.dropna().to_numpy()
            y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
            x = np.r_[neg, pos]
            fpr, tpr, _ = roc_curve(y, x)
            a.plot(fpr, tpr, color=c, ls=ls, lw=1.4, label=lab)
            aucs.append(f"{roc_auc_score(y, x):.2f}")
        a.plot([0, 1], [0, 1], color=GRID, lw=0.8)
        a.text(0.97, 0.04, "AUC " + " / ".join(aucs), transform=a.transAxes, ha="right", fontsize=6, color=INK,
               bbox=dict(fc="white", ec="none", pad=1))
        a.set_xlim(-0.02, 1.02)
        a.set_ylim(-0.02, 1.02)
        a.set_title(GROUP_EN[g], loc="left", fontsize=7.5)
        a.set_xlabel("False alarm rate")
        _clean(a)
    ax[0].set_ylabel("Detection rate")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, [f"{x}" for x in lab], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=6.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, 4)


# ------------------------------------------------------------------ Fig 5: drift-rate sweep
def fig_sweep():
    s = pd.read_csv(RES / "drift_sweep" / "drift_sweep_summary.csv")
    style = [("real", BLUE, "o", "-"), ("munfrl", GREEN, "s", "-"), ("munfrl_rtk", YELLOW, "^", "-"),
             ("semireal", ORANGE, "v", "--"), ("campus", GRAY, "D", ":")]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for g, c, mk, ls in style:
        d = s[s.group == g].sort_values("rate")
        p = d.p_before5.where(d.p_before5.notna(), d.p_detect)  # rate·window < 5 m: detection probability
        ax[0].plot(d.rate, 100 * p, color=c, marker=mk, ms=4, ls=ls, lw=1.2, label=GROUP_EN[g])
        ax[1].plot(d.rate, d.offset_med, color=c, marker=mk, ms=4, ls=ls, lw=1.2)
    ax[0].axhline(95, color=INK, lw=0.6)
    ax[0].set_ylabel("Detected before 5 m offset [%]")
    ax[1].set_ylabel("Median offset at detection [m]")
    for a in ax:
        a.set_xscale("log")
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a.set_xticks([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8])
        a.set_xticklabels(["0.05", "0.08", "0.12", "0.18", "0.25", "0.35", "0.5", "0.8"])
        a.set_xlabel("Drift rate [m/s]")
        _clean(a)
    ax[0].legend(fontsize=6.2, loc="lower right")
    fig.tight_layout()
    _save(fig, 5)


# ------------------------------------------------------------------ Fig 6: adaptive attacker
def fig_adaptive(groups=("real", "munfrl", "munfrl_rtk", "semireal", "campus", "open_field", "flat_poor")):
    s = pd.read_csv(RES / "adaptive" / "adaptive_summary.csv")
    meth = [("soft", "Proposed", BLUE), ("hard", "Hard switching", GRAY), ("alja_lio", "Alja'afreh et al. (2026)", YELLOW),
            ("gu_lio_b30", "Gu et al. (2021)", GREEN), ("ci_ins", "Tanil et al. (2018)", ORANGE),
            ("gnss_ins", "GNSS/INS monitor", "#b9b8b1")]
    fig, a = plt.subplots(figsize=(7.2, 3.0))
    w = 0.13
    x = np.arange(len(groups))
    for i, (m, lab, c) in enumerate(meth):
        v = [s[(s.group == g) & (s.method == m)].worst_add_err_mean.iloc[0] for g in groups]
        a.bar(x + (i - 2.5) * w, v, w * 0.9, color=c, label=lab)
    a.set_yscale("log")
    a.set_ylabel("Worst-case added error [m]")
    a.set_xticks(x)
    a.set_xticklabels([GROUP_EN[g].replace(" (", "\n(").replace(": ", ":\n") for g in groups], fontsize=6.5)
    a.legend(ncol=3, fontsize=6.6, loc="upper left")
    a.set_ylim(1, 400)
    _clean(a)
    a.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, 6)


if __name__ == "__main__":
    fig_architecture()
    fig_lio()
    fig_timeseries()
    fig_roc()
    fig_sweep()
    fig_adaptive()
    print("figures ->", OUT)
