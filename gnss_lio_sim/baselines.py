"""Literatürden karşılaştırma yöntemleri (yayımlanmış GNSS aldatma tespit yöntemlerinin uyarlamaları).

ci_ins : Tanil, Khanafseh, Joerger, Pervan (2018), "An INS monitor to detect GNSS spoofers capable of
         tracking vehicle position", IEEE TAES 54(1). GNSS/INS Kalman süzgecinin normalize yeniliklerinin
         birikimli (pencereli) ki-kare izleyicisi (CI monitor). LiDAR kullanılmaz. Gevşek bağlı
         uyarlamada yenilik = GNSS konum + hız yeniliği. Makale tespit sonrası toparlanmayı ele almaz
         (Kujur ve ark., NAVIGATION 2024); burada sert dışlama + önerilen yöntemdeki yeniden kabul
         kuralları kullanılır.
gu_lio : Gu, Xing, You (2021), "GNSS spoofing detection based on coupled visual/inertial/GNSS navigation
         system", Sensors 21(20):6769. GNSS konumu ile odometri konumunun farkı δ = p_GNSS − p_odo,
         l_k = Σ_{son N} δᵀ(S_GNSS + S_odo)⁻¹δ, alarm l_k ≥ β·Th (Th: χ² eşiği), alarmda GNSS ağırlığı
         α = Th²/(4 l_k²) (α < 0,01 ⇒ GNSS çıkarılır). Makaledeki görsel-ataletsel odometri yerine aynı
         LIO kullanılır; odometri ENU'ya ilk `init_time` saniyedeki GNSS ile ötelenerek hizalanır
         (makalede 60 s). S_odo makalede tanımlanmadığından önerilen yöntemin D testindeki LIO
         belirsizlik modelinin aynısı (birikimli, tarama başına sınırlı) kullanılır. β makalede
         sayısal verilmediği için geliştirme kümesinde ayarlanır.
alja_lio : Aljaafreh, Karime, Al Mallah (2026), "Integrity-aware hierarchical navigation for UAVs under
         GNSS spoofing and multi-sensor degradation", Drones 10(10):738. GNSS yeniliğinin LiDAR-ataletsel
         öngörüye göre NIS'i J (konum + hız), süreklilik sayacı c (J > γ ise artar, değilse azalır; c ≥ N_G
         alarm), tek yönlü CUSUM g = max(0, g + J − κ), κ = 0,85γ (g > h alarm). Alarmda GNSS karantinaya
         alınır; N_R ardışık tutarlı çağdan sonra yeniden kabul edilir. Makaledeki görsel kurtarma katmanı
         (Google Earth eşleştirme) kapsam dışıdır; karantina sırasında yalnız LIO-IMU ile seyredilir.
         Makale değerleri: α = 0,01, N_G = 3, κ = 0,85γ, h = 6,5, N_R = 10 (10 Hz GNSS).

    python -m gnss_lio_sim baselines tune   # parametre seçimi (geliştirme kümesi, önerilen yöntemle aynı kural)
    python -m gnss_lio_sim baselines run    # benzetim (tam) + gerçek veri karşılaştırması
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import load_config
from .detection import IntegrityMonitor

ATTACKS = ["clean", "jump", "ramp", "slow_drift", "noise", "dropout"]
SPOOF = ["jump", "ramp", "slow_drift", "noise"]
# karşılaştırmada koşturulan çalışma noktaları: (etiket, yöntem, ayar). Gu ve ark. geliştirme kümesinde
# YA ≤ %1 kısıtını ancak β ≥ 120'de sağlar ve bu noktada sıçramayı da kaçırır; ödünleşimi göstermek
# için daha duyarlı iki çalışma noktası da raporlanır (bkz. results/baselines/tune_gu_lio.csv)
VARIANTS = [("ci_ins", "ci_ins", None),
            ("gu_lio_b8", "gu_lio", {"baselines": {"gu": {"beta": 8.0}}}),
            ("gu_lio_b30", "gu_lio", {"baselines": {"gu": {"beta": 30.0}}}),
            ("gu_lio_b120", "gu_lio", {"baselines": {"gu": {"beta": 120.0}}}),
            ("alja_lio", "alja_lio", None),
            ("alja_lio_paper", "alja_lio", {"baselines": {"alja": {"alpha": 0.01, "n_g": 3, "h": 6.5}}})]
TUNE_GRID = {"ci_ins": {"window": [10, 50, 150], "alpha": [1e-3, 1e-5]},
             "gu_lio": {"beta": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0, 400.0]},
             "alja_lio": {"alpha": [1e-2, 1e-3, 1e-4], "n_g": [3, 5, 10], "h": [6.5, 25.0, 100.0, 400.0]}}
TUNE_KEY = {"ci_ins": "ci", "gu_lio": "gu", "alja_lio": "alja"}


class CIMonitor(IntegrityMonitor):
    """Tanil ve ark. (2018) CI izleyicisi: q = Σ_pencere γᵀS⁻¹γ ~ χ²(dof); sert dışlama."""

    def __init__(self, cfg):
        super().__init__(cfg.detector, cfg.transition, cfg.gnss, "hard",
                         use_velocity=cfg.fusion.use_gnss_velocity, use_disp=False)
        b = cfg.baselines.ci
        self.alpha = b.alpha
        self.win_p = deque(maxlen=b.window)
        self.win_v = deque(maxlen=b.window)

    def stats(self):
        n, m = len(self.win_p), len(self.win_v)
        if not n:
            return 0.0, 0.0, 0.0, 0.0
        q = sum(self.win_p) + sum(self.win_v)
        return q / chi2.ppf(1 - self.alpha, self.k * (n + m)), 0.0, 0.0, 0.0


class AljaafrehMonitor:
    """Aljaafreh ve ark. (2026): NIS + süreklilik + CUSUM, GNSS karantinası ve N_R çağlık yeniden kabul."""

    def __init__(self, cfg):
        b = cfg.baselines.alja
        self.d = cfg.detector
        self.horiz = bool(cfg.detector.get("horizontal_only", False))
        self.k = 2 if self.horiz else 3
        self.m = self.k * (2 if cfg.fusion.use_gnss_velocity else 1)
        self.gamma = chi2.ppf(1 - b.alpha, self.m)
        self.kappa, self.h = b.kappa * self.gamma, b.h
        self.n_g, self.n_r, self.c_max = b.n_g, b.n_r, b.c_max
        self.c, self.g, self.ok = 0, 0.0, 0
        self.quarantined = False
        self.last_valid_t = None
        self.J = 0.0

    def _info(self, dropout=False, alarm=False):
        r = max(self.c / self.n_g, self.g / self.h)
        return {"rA": self.J / self.gamma, "rB": self.g / self.h, "rC": 0.0, "rD": 0.0, "rho": r,
                "dropout": dropout, "alarm": bool(alarm or dropout)}

    def step(self, t, nis_p, nis_v, *_, **__):
        if nis_p is None:
            dropout = self.last_valid_t is not None and t - self.last_valid_t > self.d.dropout_timeout
            return np.inf, self._info(dropout)
        self.last_valid_t = t
        J = nis_p + (nis_v if nis_v is not None else 0.0)
        self.J = J
        self.c = min(self.c + 1, self.c_max) if J > self.gamma else max(self.c - 1, 0)
        self.g = max(0.0, self.g + J - self.kappa)
        alarm = self.c >= self.n_g or self.g > self.h
        if not self.quarantined:
            if alarm:
                self.quarantined, self.ok = True, 0
        else:
            self.ok = self.ok + 1 if J <= self.gamma else 0
            if self.ok >= self.n_r:  # yeniden kabul: istatistikler sıfırlanır (makaledeki gibi)
                self.quarantined, self.c, self.g = False, 0, 0.0
        # alarm kaydı sert anahtarlamadaki gibi: karantina süresince GNSS reddedilmiş sayılır
        return (np.inf if self.quarantined else 1.0), self._info(alarm=alarm or self.quarantined)


class GuMonitor:
    """Gu ve ark. (2021): GNSS − odometri mutlak konum farkının pencereli ki-kare testi."""

    def __init__(self, cfg, Rp):
        b = cfg.baselines.gu
        self.d = cfg.detector
        self.horiz = bool(cfg.detector.get("horizontal_only", False))
        self.k = 2 if self.horiz else 3
        self.pf, self.beta, self.period, self.init = b.pf, b.beta, b.period, b.init_time
        self.buf = deque(maxlen=b.window)
        self.Rp = Rp
        self.t0 = self.offset = self.C0 = self.next_t = self.last_valid_t = None
        self.init_res = []
        self.S_align = np.zeros((3, 3))
        self.lam, self.ratio, self.alarm = 1.0, 0.0, False
        self.excluded = False

    def _info(self, dropout=False):
        r = self.ratio
        return {"rA": r, "rB": 0.0, "rC": 0.0, "rD": 0.0, "rho": r, "dropout": dropout,
                "alarm": bool(self.alarm or dropout)}

    def step(self, t, nis_p, nis_v, gnss_pos=None, D=None, C=None, counts=(0, 0), p_lio=None, C_abs=None):
        if nis_p is None:
            dropout = self.last_valid_t is not None and t - self.last_valid_t > self.d.dropout_timeout
            return np.inf, self._info(dropout)
        self.last_valid_t = t
        if p_lio is None:
            return 1.0, self._info()
        if self.t0 is None:
            self.t0 = t
        if self.offset is None:
            if t < self.t0 + self.init:
                self.init_res.append(gnss_pos - p_lio)  # hizalama dönemi: GNSS sağlıklı varsayılır
                return 1.0, self._info()
            n = len(self.init_res)
            self.offset = np.mean(self.init_res, axis=0) if n else np.zeros(3)
            self.S_align = self.Rp / n if n else np.zeros((3, 3))
            self.C0 = C_abs.copy()
        if self.next_t is None or t >= self.next_t - 1e-9:
            self.next_t = t + self.period
            delta = gnss_pos - (p_lio + self.offset)
            S = self.Rp + self.S_align + (C_abs - self.C0)
            k = self.k
            self.buf.append(float(delta[:k] @ np.linalg.solve(S[:k, :k], delta[:k])))
            th = chi2.ppf(1 - self.pf, k * len(self.buf))
            l = sum(self.buf)
            self.ratio = l / (self.beta * th)
            self.alarm = self.ratio >= 1.0
            if self.alarm:
                a = th ** 2 / (4.0 * l ** 2)
                self.lam = np.inf if a < 0.01 else 1.0 / a
            else:
                self.lam = 1.0
        self.excluded = not np.isfinite(self.lam)
        return self.lam, self._info()


# ------------------------------------------------------------------ deney
def _run(args):
    from .fusion import run_fusion
    from .metrics import eval_run
    from .sim.flight import load_flight
    from .spoofing import inject
    cfg_path, overrides, path, group, method, attacks, label = args
    cfg = load_config(cfg_path, overrides)
    fl = load_flight(path)
    meta = fl["meta"]
    out = []
    for a in attacks:
        att = inject(fl["gnss_t"], fl["gnss_pos"], fl["gnss_vel"], meta, a, cfg)
        m = eval_run(fl, att, run_fusion(fl, att, method, cfg), cfg)
        out.append({"group": group or meta["env"], "id": meta["id"], "env": meta["env"], "attack": a,
                    "method": label, "params": str(overrides.get("baselines", "")) if overrides else "", **m})
    return out


def _pool(jobs, workers):
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_run, jobs, chunksize=1):
            rows += r
    return pd.DataFrame(rows)


def _far(d):
    return d.fa_epochs.sum() / max(d.healthy_epochs.sum(), 1)


def tune(flights_dir, quick_overrides, out_dir, workers, far_max=0.01, methods=None, log=print):
    """Önerilen yöntemin ayarındaki kuralın aynısı: geliştirme kümesinde YA ≤ far_max kısıtı altında
    yavaş sürüklemeyi 5 m'den önce tespit oranı en yüksek (eşitlikte YA en düşük) parametre."""
    flights = sorted(Path(flights_dir).glob("*.npz"))
    chosen = {}
    for method, grid in TUNE_GRID.items():
        if methods and method not in methods:
            continue
        key = TUNE_KEY[method]
        combos = [dict(zip(grid, v)) for v in product(*grid.values())]
        jobs = []
        for c in combos:
            ov = {**quick_overrides, "baselines": {key: c}}
            jobs += [(None, ov, str(p), None, method, ["clean", "slow_drift", "jump"], method) for p in flights]
        df = _pool(jobs, workers)
        rows = []
        for params, g in df.groupby("params"):
            sd = g[g.attack == "slow_drift"]
            rows.append({"method": method, "params": params, "far": _far(g),
                         "slow_before_5m": float((sd.detected & (sd.offset_at_det < 5)).mean()),
                         "jump_delay": g[g.attack == "jump"].delay.mean()})
        s = pd.DataFrame(rows)
        ok = s[s.far <= far_max]
        best = (ok if len(ok) else s).sort_values(["slow_before_5m", "far"], ascending=[False, True]).iloc[0]
        log(s.round(4).to_string(index=False))
        log(f"[baselines tune] {method}: {best.params} (YA {100 * best.far:.2f}%, "
            f"5 m öncesi {100 * best.slow_before_5m:.0f}%, {'kısıt sağlandı' if len(ok) else 'kısıt sağlanamadı'})")
        s.to_csv(Path(out_dir) / f"tune_{method}.csv", index=False)
        chosen[method] = best.params
    return chosen


def run(jobs, out_dir, workers, methods=None, groups=None):
    """jobs: [(cfg_path, flight_path, group)] — her uçuş × VARIANTS × ATTACKS. methods / groups verilirse
    yalnız o yöntemler / gruplar koşturulur ve önceki sonuç dosyasındaki satırlarının yerine yazılır."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v for v in VARIANTS if not methods or v[1] in methods]
    df = _pool([(c, ov, str(p), g, m, ATTACKS, lab) for c, p, g in jobs for lab, m, ov in variants], workers)
    f = out / "baselines_results.csv"
    if (methods or groups) and f.exists():
        old = pd.read_csv(f)
        rerun = old.method.isin([v[0] for v in variants]) & (old.group.isin(groups) if groups else True)
        df = pd.concat([old[~rerun], df], ignore_index=True)
    df.to_csv(f, index=False)
    return df


def summarize(df):
    """Yöntem × grup özet: YA, sıçrama tespit gecikmesi, yavaş sürükleme 5 m öncesi tespit,
    saldırılarda ek hata ve adım süreksizliği, GNSS kesintisinde sürüklenme."""
    rows = []
    for (g, m), d in df.groupby(["group", "method"]):
        sd, jd = d[d.attack == "slow_drift"], d[d.attack == "jump"]
        sp = d[d.attack.isin(SPOOF)]
        rows.append({"group": g, "method": m, "n_runs": len(d),
                     "far_pct": 100 * _far(d),
                     "far_run_pct": 100 * d[d.attack == "clean"].any_false_alarm.astype(float).mean(),
                     "jump_det_pct": 100 * jd.detected.mean(), "jump_delay_s": jd.delay.median(),
                     "slow_before5_pct": 100 * float((sd.detected & (sd.offset_at_det < 5)).mean()),
                     "ramp_det_pct": 100 * d[d.attack == "ramp"].detected.mean(),
                     "peak_add_err_m": sp.peak_add_err.mean(), "max_peak_add_err_m": sp.peak_add_err.max(),
                     "jump_post_m": sp.jump_post.mean(),
                     "dropout_drift_pct": d[d.attack == "dropout"].drift_pct.mean()})
    return pd.DataFrame(rows)


LABELS = {"gnss_ins": "GNSS/INS (dışlamasız)", "ci_ins": "Tanil ve ark. 2018 (CI, GNSS/INS)",
          "gu_lio_b8": "Gu ve ark. 2021, β = 8", "gu_lio_b30": "Gu ve ark. 2021, β = 30",
          "gu_lio_b120": "Gu ve ark. 2021, β = 120",
          "alja_lio": "Aljaafreh ve ark. 2026 (ayarlı)", "alja_lio_paper": "Aljaafreh ve ark. 2026 (makale değerleri)",
          "hard": "Sert anahtarlama (ablasyon)",
          "soft": "Önerilen yöntem"}
GROUPS = {"real": "Gerçek veri (MARS-LVIG)", "munfrl": "Gerçek veri (MUN-FRL, tek nokta GNSS)", "munfrl_rtk": "Gerçek veri (MUN-FRL, RTK GNSS)", "semireal": "Yarı-gerçek (AMtown03)", "campus": "Benzetim: kampüs",
          "open_field": "Benzetim: açık arazi", "flat_poor": "Benzetim: yapısız düz alan"}


def compare(root, out_dir):
    """Önerilen yöntem ve ablasyonlar (ana deney sonuçları) + literatür yöntemleri: tek özet tablo."""
    root, out = Path(root), Path(out_dir)
    parts = [pd.read_csv(out / "baselines_results.csv")]
    for f, group in ((root / "results" / "full" / "results.csv", None),
                     (root / "results" / "real" / "results.csv", "real"),
                     (root / "results" / "semireal" / "results.csv", "semireal"),
                     (root / "results" / "munfrl" / "results.csv", "munfrl"),
                     (root / "results" / "munfrl_rtk" / "results.csv", "munfrl_rtk")):
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d.severity == 1.0) & d.method.isin(["gnss_ins", "hard", "soft"])].copy()
        d["group"] = group or d.env
        parts.append(d)
    s = summarize(pd.concat(parts, ignore_index=True))
    s["order"] = s.method.map({m: i for i, m in enumerate(LABELS)})
    s = s.sort_values(["group", "order"]).drop(columns="order")
    s.to_csv(out / "comparison.csv", index=False)
    plot(s, out / "comparison.png")
    return s


def plot(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, base, accent = "#2b2b29", "#6b6a64", "#e4e3de", "#a9a8a0", "#2a78d6"
    plt.rcParams.update({"axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": ink, "font.size": 9})
    groups = [g for g in GROUPS if g in set(s.group)]
    methods = [m for m in LABELS if m in set(s.method)]
    fig, ax = plt.subplots(3, len(groups), figsize=(3.1 * len(groups) + 2.2, 8.0), sharey=True)
    rows = [("peak_add_err_m", "Saldırılarda ek hata [m]\n(ortalama, log)"),
            ("slow_before5_pct", "Yavaş sürükleme 5 m'den\nönce tespit [%]"),
            ("far_pct", "Yanlış alarm oranı [%]\n(sağlıklı GNSS çağları)")]
    y = np.arange(len(methods))[::-1]
    for j, g in enumerate(groups):
        d = s[s.group == g].set_index("method").reindex(methods)
        for i, (col, lab) in enumerate(rows):
            a = ax[i, j]
            v = d[col].to_numpy(float)
            colors = [accent if m == "soft" else base for m in methods]
            a.barh(y, np.nan_to_num(v), height=0.62, color=colors, edgecolor="#fcfcfb", linewidth=1)
            for yy, vv in zip(y, v):
                if np.isfinite(vv):
                    a.text(vv * (1.08 if col == "peak_add_err_m" else 1) + (0 if col == "peak_add_err_m" else 1 if col == "far_pct" else 2),
                           yy, f"{vv:.0f}" if col == "slow_before5_pct" else f"{vv:.1f}",
                           va="center", fontsize=7.5, color=ink)
            if col == "peak_add_err_m":
                a.set_xscale("log")
                a.set_xlim(0.3, max(np.nanmax(s[col]) * 4, 10))
            elif col == "far_pct":
                a.set_xlim(0, 60)
            else:
                a.set_xlim(0, 118)
            a.grid(axis="x", color=grid, lw=0.8)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if i == 0:
                a.set_title(GROUPS[g], color=ink, loc="left", fontsize=9.5)
            if j == 0:
                a.set_ylabel(lab)
    ax[0, 0].set_yticks(y, [LABELS[m] for m in methods])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
