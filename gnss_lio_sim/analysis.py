"""Sonuç analizi: hipotez testleri (H1-H3, Hedef 4), ROC, ANOVA, regresyon, grafikler ve rapor."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# Doğrulanmış kategorik palet (sabit sıra; renk varlığı izler, sıralamayı değil)
METHOD_COLORS = {
    "hard": "#2a78d6", "voting": "#eb6834", "soft": "#1baf7a",
    "gnss_ins": "#eda100", "monitor": "#e87ba4", "lio_only": "#008300",
}
METHOD_LABELS = {
    "hard": "Sert anahtarlama", "voting": "Sabit eşikli oylama", "soft": "Sürekli ağırlıklandırma (önerilen)",
    "gnss_ins": "GNSS/INS (LIO yok)", "monitor": "İzleme (dışlama yok)", "lio_only": "Yalnız LIO",
}
ENV_COLORS = {"open_field": "#2a78d6", "campus": "#eb6834", "flat_poor": "#1baf7a"}
ENV_LABELS = {"open_field": "Açık arazi", "campus": "Yarı yapılı kampüs", "flat_poor": "Yapısal olarak fakir düz alan"}
ATTACK_LABELS = {"jump": "Ani sıçrama", "ramp": "Sabit kayma", "slow_drift": "Yavaş sürükleme",
                 "noise": "Gürültü artışı", "dropout": "Tam sinyal kaybı", "clean": "Sağlıklı"}
SPOOF = ["jump", "ramp", "slow_drift", "noise"]
CMP_METHODS = ["hard", "voting", "soft"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2.0, "figure.dpi": 110,
    })


def _ms(x):
    x = pd.Series(x).dropna()
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}" if len(x) > 1 else (f"{x.mean():.2f}" if len(x) else "—")


def _md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ ROC
ROC_DEADLINE = {"jump": lambda c: 3.0, "ramp": lambda c: 5.0 / c.attacks.ramp.rate,
                "slow_drift": lambda c: 5.0 / c.attacks.slow_drift.rate, "noise": lambda c: 10.0}


def roc_curves(runs_dir, cfg, thresholds):
    healthy, first = [], {a: [] for a in ROC_DEADLINE}
    for f in sorted(Path(runs_dir).glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        gt = d["g_t"]
        rho = d["clean/monitor/rho"]
        healthy.append(rho[gt > 5.0])
        t_on = meta["onset"]
        for a in ROC_DEADLINE:
            r = d[f"{a}/monitor/rho"]
            w = (gt >= t_on) & (gt <= t_on + ROC_DEADLINE[a](cfg))
            first[a].append(r[w].max() if w.any() else 0.0)
    healthy = np.concatenate(healthy)
    pfa = np.array([(healthy >= c).mean() for c in thresholds])
    pd_ = {a: np.array([(np.array(v) >= c).mean() for c in thresholds]) for a, v in first.items()}
    return pfa, pd_


# ------------------------------------------------------------------ ana analiz
def analyze(out_dir, cfg, log=print):
    _style()
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    df = pd.read_csv(out_dir / "results.csv")
    seg = pd.read_csv(out_dir / "drift_segments.csv")
    main = df[df.severity == 1.0]
    rep = {}
    md = []

    n_flights = df["id"].nunique()
    md.append("# GNSS Aldatma Koşullarında LiDAR-Ataletsel Füzyon — Benzetim Sonuç Raporu\n")
    md.append(f"Uçuş sayısı: **{n_flights}** · koşu sayısı: **{len(df)}** · ortamlar: "
              f"{', '.join(ENV_LABELS[e] for e in df.env.unique())} · yörüngeler: "
              f"{', '.join(df.traj.unique())} · hızlar: {', '.join(f'{s:g}' for s in sorted(df.speed.unique()))} m/s\n")
    md.append("Tüm aldatma senaryoları gerçek RF yayını olmaksızın, sağlıklı uçuş kayıtlarındaki GNSS "
              "ölçümlerine yazılım düzeyinde enjekte edilmiştir (kayıt-üzerinde-oynatma). Referans "
              "yörünge, deney konusu GNSS kanalından bağımsız RTK benzetimidir.\n")

    # ---------------- Hedef 1: LIO taban hattı
    lio = main[main.method == "lio_only"]
    t1 = lio.groupby("env").agg(
        ATE_hizali=("ate_aligned", _ms), ATE_ham=("rmse", _ms), RPE_1s=("rpe_1s", _ms),
        Suruklenme_60s_yuzde=("drift_pct", _ms), Islem_ms_ort=("lio_proc_ms_mean", "mean"),
        Islem_ms_p95=("lio_proc_ms_p95", "mean")).reset_index()
    t1["env"] = t1.env.map(ENV_LABELS)
    t1 = t1.round(2)
    md.append("## 1. LIO taban hattı (Hedef 1, İP1)\n")
    md.append("GNSS füzyonsuz LiDAR-ataletsel odometri; ATE/RPE [m], sürüklenme: aldatma penceresine "
              "denk gelen 60 s'lik bölümde kat edilen mesafeye oranla [%].\n")
    md.append(_md_table(t1) + "\n")
    rep["lio_rt_ok"] = bool(lio.lio_proc_ms_p95.max() < 1000.0 / cfg.lidar.rate)

    # ---------------- H1: tespit
    det = main[main.method.isin(CMP_METHODS) & main.attack.isin(ATTACK_LABELS) & (main.attack != "clean")]
    rows = []
    for a in ["jump", "ramp", "slow_drift", "noise", "dropout"]:
        for m in CMP_METHODS:
            s = det[(det.attack == a) & (det.method == m)]
            if s.empty:
                continue
            rows.append({"Saldırı": ATTACK_LABELS[a], "Yöntem": METHOD_LABELS[m],
                         "Tespit oranı": f"{100 * s.detected.mean():.0f}%",
                         "Gecikme [s]": _ms(s.delay), "Tespitte öteleme [m]": _ms(s.offset_at_det),
                         "Tespitte konum hatası [m]": _ms(s.err_at_det)})
    far = main[main.method.isin(CMP_METHODS)].groupby("method").apply(
        lambda s: s.fa_epochs.sum() / max(s.healthy_epochs.sum(), 1), include_groups=False)
    far_run = main[(main.attack == "clean") & main.method.isin(CMP_METHODS)].groupby("method").any_false_alarm.mean()
    md.append("## 2. Aldatma tespiti (H1, Hedef 2, İP2)\n")
    md.append(_md_table(pd.DataFrame(rows)) + "\n")
    md.append("Yanlış alarm oranı (sağlıklı GNSS çağları; tüm temiz koşular + saldırı öncesi bölümler):\n")
    md.append(_md_table(pd.DataFrame({
        "Yöntem": [METHOD_LABELS[m] for m in far.index],
        "Çağ bazında YA oranı": [f"{100 * v:.2f}%" for v in far.values],
        "En az bir YA içeren temiz uçuş": [f"{100 * far_run.get(m, np.nan):.0f}%" for m in far.index]})) + "\n")
    # yavaş sürükleme: ortam bazında (LIO doğrulamasının geometriye bağımlılığı)
    sl = det[det.attack == "slow_drift"].copy()
    sl["before5"] = (sl.offset_at_det < 5.0) & sl.detected
    env_tab = sl.groupby(["env", "method"]).agg(
        Tespit_5m_once=("before5", lambda x: f"{100 * x.mean():.0f}%"),
        Gecikme_s=("delay", _ms), Oteleme_m=("offset_at_det", _ms)).reset_index()
    env_tab["env"] = env_tab.env.map(ENV_LABELS)
    env_tab["method"] = env_tab.method.map(METHOD_LABELS)
    env_tab.columns = ["Ortam", "Yöntem", "5 m öncesi tespit", "Gecikme [s]", "Tespitte öteleme [m]"]
    md.append("Yavaş sürüklemeli aldatmanın ortam bazında tespiti (LIO–GNSS yer değiştirme testinin "
              "kullanılabilirliği ortam geometrisine bağlıdır):\n")
    md.append(_md_table(env_tab) + "\n")
    soft = det[det.method == "soft"]
    jd = soft[soft.attack == "jump"]
    sd = soft[soft.attack == "slow_drift"]
    rep["H1_slow_before5m_by_env"] = sl[sl.method == "soft"].groupby("env").before5.mean().to_dict()
    rep["H1_far"] = float(far.get("soft", np.nan))
    rep["H1_jump_delay"] = float(jd.delay.mean())
    rep["H1_jump_detect_rate"] = float(jd.detected.mean())
    rep["H1_slow_before5m"] = float(((sd.offset_at_det < 5.0) & sd.detected).mean())
    h1_ok = rep["H1_far"] < 0.02 and rep["H1_jump_delay"] < 3.0 and rep["H1_slow_before5m"] >= 0.95

    # ROC
    thr = np.logspace(-1.3, 2.0, 80)
    pfa, pdd = roc_curves(out_dir / "runs", cfg, thr)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    roc_cols = {"jump": "#2a78d6", "ramp": "#eb6834", "slow_drift": "#1baf7a", "noise": "#eda100"}
    i1 = np.argmin(np.abs(thr - 1.0))
    for a, y in pdd.items():
        ax.plot(np.maximum(pfa, 1e-5), y, color=roc_cols[a], label=ATTACK_LABELS[a])
        ax.plot(max(pfa[i1], 1e-5), y[i1], "o", ms=8, color=roc_cols[a], mec=SURFACE, mew=2)
    ax.axvline(0.02, color=INK2, lw=1, ls="--")
    ax.text(0.023, 0.5, "%2 YA\nhedefi", color=INK2, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Yanlış alarm olasılığı (GNSS çağı başına)")
    ax.set_ylabel("Tespit olasılığı (süre sınırı içinde)")
    ax.set_title("ROC — birleşik tutarlılık istatistiği ρ (izleme kipi)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc.png")
    plt.close(fig)
    md.append("![ROC](figures/roc.png)\n")
    md.append("ROC eğrileri, GNSS'in her zaman tam ağırlıkla füzyona katıldığı izleme kipindeki (en kötü durum: "
              "aldatmanın kestirime emilmesi) ρ = max(χ²-pencere, CUSUM, hız-χ²) istatistiği üzerinde eşik "
              "taranarak çıkarılmıştır (gözlenmeyen YA olasılığı grafiğin sol kenarında 1e-5 olarak çizilir; "
              "Pd = 1 olan eğriler üst üste biner). Süre sınırları: ani sıçrama 3 s; kayma/sürükleme türlerinde "
              "kasıtlı ötelemenin 5 m'ye ulaştığı an; gürültü artışı 10 s. Noktalar çalışma eşiğini (ρ=1) gösterir.\n")

    # gecikme grafiği
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=False)
    for ax, a in zip(axes, SPOOF):
        data = [det[(det.attack == a) & (det.method == m)].delay.dropna().values for m in CMP_METHODS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, m in zip(bp["boxes"], CMP_METHODS):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_edgecolor(SURFACE)
        for el in bp["medians"]:
            el.set_color(INK)
        ax.set_xticks([1, 2, 3], ["Sert", "Oylama", "Sürekli"])
        ax.set_title(ATTACK_LABELS[a])
        ax.set_ylabel("Tespit gecikmesi [s]" if a == "jump" else "")
    fig.tight_layout()
    fig.savefig(fig_dir / "detection_delay.png")
    plt.close(fig)
    md.append("![Tespit gecikmesi](figures/detection_delay.png)\n")

    # ---------------- H2: geçiş
    tr = main[main.method.isin(CMP_METHODS + ["monitor", "gnss_ins"]) & main.attack.isin(SPOOF)]
    rows = []
    for a in SPOOF:
        for m in ["gnss_ins", "monitor"] + CMP_METHODS:
            s = tr[(tr.attack == a) & (tr.method == m)]
            rows.append({"Saldırı": ATTACK_LABELS[a], "Yöntem": METHOD_LABELS[m],
                         "Ek hata sıçraması [m]": _ms(s.peak_add_err),
                         "Saldırı penceresi [m]": _ms(s.peak_add_err_attack),
                         "Geri kabul [m]": _ms(s.peak_add_err_recovery),
                         "Adım süreksizliği [m]": _ms(s.jump_post),
                         "RMSE (saldırı sonrası) [m]": _ms(s.rmse_post)})
    md.append("## 3. Sarsımsız kaynak geçişi (H2, Hedef 3, İP3)\n")
    md.append("*Ek hata sıçraması*: saldırı başlangıcından uçuş sonuna kadarki en büyük konum hatası − saldırı "
              "öncesi 10 s ortalama hata. *Adım süreksizliği*: 10 Hz çıktıda kestirim artımı ile gerçek "
              "artım arasındaki en büyük fark (denetleyiciye yansıyan sıçrama).\n")
    md.append(_md_table(pd.DataFrame(rows)) + "\n")
    piv = tr[tr.method.isin(["hard", "soft"])].pivot_table(
        index=["id", "attack"], columns="method", values="peak_add_err")
    piv_j = tr[tr.method.isin(["hard", "soft"])].pivot_table(
        index=["id", "attack"], columns="method", values="jump_post")
    # H2 birincil ölçüt: geçiş anı yörünge sıçraması = kestirimdeki adım süreksizliği
    # (öneri 1.1: "sert anahtarlama ... denetleyiciye yansıyan yörünge sıçramalarına yol açar")
    red_j = 1 - piv_j.soft.mean() / piv_j.hard.mean()
    red = 1 - piv.soft.mean() / piv.hard.mean()
    w = stats.wilcoxon(piv_j.hard, piv_j.soft) if len(piv_j) > 5 else None
    rep["H2_reduction"] = float(red_j)
    rep["H2_peak_err_reduction"] = float(red)
    rep["H2_p"] = float(w.pvalue) if w is not None else np.nan
    per_attack = {a: 1 - piv_j.xs(a, level="attack").soft.mean() / piv_j.xs(a, level="attack").hard.mean()
                  for a in SPOOF}
    md.append(f"Sürekli ağırlıklandırmanın sert anahtarlamaya göre **geçiş anı yörünge sıçramasında (adım "
              f"süreksizliği) ortalama azalma: {100 * red_j:.1f}%** (eşleştirilmiş Wilcoxon p = {rep['H2_p']:.2g}). "
              "Saldırı türüne göre: "
              + ", ".join(f"{ATTACK_LABELS[a]} {100 * v:.0f}%" for a, v in per_attack.items())
              + f". İkincil ölçüt olarak ek hata sıçramasındaki (tepe hata) değişim: {100 * red:.1f}%.\n")
    per_env = {}
    for e in piv_j.reset_index().merge(tr[["id", "env"]].drop_duplicates(), on="id").env.unique():
        ids = tr[tr.env == e].id.unique()
        sub = piv_j[piv_j.index.get_level_values("id").isin(ids)]
        per_env[e] = 1 - sub.soft.mean() / sub.hard.mean()
    rep["H2_reduction_by_env"] = per_env
    md.append("Ortam bazında yörünge sıçramasındaki azalma (ek analiz): "
              + ", ".join(f"{ENV_LABELS[e]} {100 * v:.0f}%" for e, v in per_env.items()) + ".\n")

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(SPOOF))
    bw = 0.26
    for j, m in enumerate(CMP_METHODS):
        vals = [tr[(tr.attack == a) & (tr.method == m)].jump_post.mean() for a in SPOOF]
        errs = [tr[(tr.attack == a) & (tr.method == m)].jump_post.std() for a in SPOOF]
        yerr = [np.minimum(vals, errs), errs]  # alt çubuk sıfırda kırpılır (negatif olamaz)
        ax.bar(x + (j - 1) * (bw + 0.02), vals, bw, yerr=yerr, color=METHOD_COLORS[m],
               label=METHOD_LABELS[m], error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    ax.set_xticks(x, [ATTACK_LABELS[a] for a in SPOOF])
    ax.set_ylabel("Adım süreksizliği [m]")
    ax.set_title("Geçiş anı yörünge sıçraması (ortalama ± std)")
    ax.legend(ncols=3, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "transition_error.png")
    plt.close(fig)
    md.append("![Geçiş hatası](figures/transition_error.png)\n")

    # ANOVA
    md.append("### İki yönlü ANOVA — geçiş anı yörünge sıçraması ~ yöntem × ortam (aldatma senaryoları)\n")
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        sub = tr[tr.method.isin(CMP_METHODS)]
        model = ols("jump_post ~ C(method) * C(env)", data=sub).fit()
        an = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "Etken"})
        an = an.round(4)
        md.append(_md_table(an.fillna("")) + "\n")
    except Exception as e:  # pragma: no cover
        md.append(f"ANOVA hesaplanamadı: {e}\n")

    # ---------------- Hedef 4: GNSS'siz seyir
    dr = main[(main.attack == "dropout") & main.method.isin(["soft", "lio_only", "gnss_ins"])]
    lio_dr = main[main.method == "lio_only"].assign(method="lio_only")
    t4 = pd.concat([dr[dr.method != "lio_only"], lio_dr])
    t4s = t4.groupby(["env", "method"]).agg(
        surukl=("drift_pct", _ms), surukl_m=("drift_abs", _ms), mesafe=("denied_dist", "mean")).reset_index()
    t4s["env"] = t4s.env.map(ENV_LABELS)
    t4s["method"] = t4s.method.map(METHOD_LABELS)
    t4s.columns = ["Ortam", "Yöntem", "Sürüklenme [%]", "Sürüklenme [m]", "Kat edilen mesafe [m]"]
    md.append(f"## 4. GNSS'siz seyir sürüklenmesi (Hedef 4, İP4)\n")
    md.append(f"Tam sinyal kaybı senaryosu, {cfg.flight.gnss_denied_duration:g} s GNSS'siz seyir. Hedef: "
              "sürüklenme < kat edilen mesafenin %2'si.\n")
    md.append(_md_table(t4s.round(1)) + "\n")
    soft_dr = dr[dr.method == "soft"]
    env_drift = soft_dr.groupby("env").drift_pct.mean()
    rep["H4_drift_by_env"] = env_drift.to_dict()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    envs = [e for e in ENV_LABELS if e in env_drift.index]
    for j, m in enumerate(["lio_only", "soft"]):
        s = t4[t4.method == m]
        vals = [s[s.env == e].drift_pct.mean() for e in envs]
        ax.bar(np.arange(len(envs)) + (j - 0.5) * 0.4, vals, 0.38, color=METHOD_COLORS[m],
               label=METHOD_LABELS[m])
    ax.axhline(2.0, color=INK2, lw=1, ls="--")
    ax.text(len(envs) - 0.5, 2.2, "%2 hedefi", color=INK2, ha="right", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(envs)), [ENV_LABELS[e] for e in envs])
    ax.set_ylabel("Sürüklenme / mesafe [%]")
    ax.set_title(f"{cfg.flight.gnss_denied_duration:g} s GNSS'siz seyirde sürüklenme")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "denied_drift.png")
    plt.close(fig)
    md.append("![GNSS'siz sürüklenme](figures/denied_drift.png)\n")
    ad = main[(main.attack == "dropout") & (main.method == "soft")]
    try:
        from statsmodels.formula.api import ols
        import statsmodels.api as sm
        an2 = sm.stats.anova_lm(ols("np.log10(drift_pct) ~ C(env) + C(traj) + C(speed)", data=ad).fit(), typ=2)
        md.append("ANOVA — log10(sürüklenme %) ~ ortam + yörünge + hız:\n")
        md.append(_md_table(an2.round(4).reset_index().rename(columns={"index": "Etken"}).fillna("")) + "\n")
    except Exception as e:  # pragma: no cover
        md.append(f"ANOVA hesaplanamadı: {e}\n")

    # ---------------- H3: sürüklenme-geometri regresyonu
    md.append("## 5. Sürüklenme–ortam geometrisi ilişkisi (H3, Hedef 5, İP5)\n")
    rep.update(_regression(seg, fig_dir, md, cfg))

    # ---------------- gerçek zamanlılık
    lio_p95 = lio.lio_proc_ms_p95.mean()
    fus_ms = main[main.method == "soft"].lio_update_ms.mean()
    rep["latency_ms"] = float(lio_p95 + fus_ms)
    md.append("## 6. Hesaplama bütçesi\n")
    md.append(f"LIO tarama işleme süresi: ortalama {lio.lio_proc_ms_mean.mean():.1f} ms, p95 {lio_p95:.1f} ms, "
              f"en çok {lio.lio_proc_ms_max.max():.1f} ms (tarama periyodu {1000 / cfg.lidar.rate:.0f} ms). "
              f"Füzyon LIO-güncelleme adımı: {fus_ms:.2f} ms. Uçtan uca (p95) ≈ **{rep['latency_ms']:.1f} ms** "
              "(hedef < 100 ms). Ölçümler masaüstü CPU'da tek iş parçacıklı Python içindir; Jetson üzerindeki "
              "C++ gerçeklemesi için yalnızca göreli bütçe göstergesidir.\n")

    # ---------------- örnek zaman serisi
    _example_plot(out_dir, fig_dir, md)

    # ---------------- özet
    summary = [
        ("H1 — YA oranı < %2 (önerilen yöntem)", f"{100 * rep['H1_far']:.2f}%", rep["H1_far"] < 0.02),
        ("H1 — ani sıçramada ort. tespit gecikmesi < 3 s", f"{rep['H1_jump_delay']:.2f} s", rep["H1_jump_delay"] < 3),
        ("H1 — yavaş sürükleme 5 m öteleme aşılmadan tespit (tüm ortamlar)",
         f"{100 * rep['H1_slow_before5m']:.0f}% koşu", rep["H1_slow_before5m"] >= 0.95),
    ] + [
        (f"H1 — yavaş sürükleme 5 m öncesi tespit ({ENV_LABELS[e]})", f"{100 * v:.0f}% koşu", v >= 0.95)
        for e, v in rep["H1_slow_before5m_by_env"].items()
    ] + [
        ("H2 — geçiş anı yörünge sıçramasında ≥ %50 azalma (sürekli vs sert, tüm ortamlar)",
         f"{100 * rep['H2_reduction']:.1f}%", rep["H2_reduction"] >= 0.5),
    ] + [
        (f"H2 — yörünge sıçramasında ≥ %50 azalma ({ENV_LABELS[e]})", f"{100 * v:.1f}%", v >= 0.5)
        for e, v in rep["H2_reduction_by_env"].items()
    ] + [
        ("H3 — sürüklenme-geometri modeli (çapraz doğrulama R²)", f"{rep['H3_cv_r2']:.2f}", rep["H3_cv_r2"] >= 0.5),
        ("Hedef 4 — GNSS'siz 60 s'de sürüklenme < %2 (açık arazi)",
         f"{rep['H4_drift_by_env'].get('open_field', np.nan):.2f}%", rep["H4_drift_by_env"].get("open_field", 99) < 2),
        ("Hedef 4 — GNSS'siz 60 s'de sürüklenme < %2 (kampüs)",
         f"{rep['H4_drift_by_env'].get('campus', np.nan):.2f}%", rep["H4_drift_by_env"].get("campus", 99) < 2),
        ("Hedef 4 — GNSS'siz 60 s'de sürüklenme < %2 (düz alan)",
         f"{rep['H4_drift_by_env'].get('flat_poor', np.nan):.2f}%", rep["H4_drift_by_env"].get("flat_poor", 99) < 2),
        ("Uçtan uca gecikme < 100 ms", f"{rep['latency_ms']:.1f} ms", rep["latency_ms"] < 100),
    ]
    head = ["## Özet — hipotez ve hedeflerin benzetim sonuçları\n",
            _md_table(pd.DataFrame([{"Ölçüt": a, "Sonuç": b, "Durum": "✅ sağlandı" if c else "❌ sağlanmadı"}
                                    for a, b, c in summary])) + "\n"]
    text = md[:3] + head + md[3:]
    (out_dir / "report.md").write_text("\n".join(text), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    log(f"[analyze] rapor -> {out_dir / 'report.md'}")
    for a, b, c in summary:
        log(f"   {'OK ' if c else 'X  '} {a}: {b}")
    return rep


def _regression(seg, fig_dir, md, cfg):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    seg = seg[(seg.drift_per_m > 0) & np.isfinite(seg.drift_per_m)].copy()
    seg["y"] = np.log10(seg.drift_per_m)
    seg["log_pos_info"] = np.log10(seg.pos_info_min + 1.0)
    seg["log_rot_info"] = np.log10(seg.rot_info_min + 1.0)
    seg["log_n_vert"] = np.log10(seg.n_vert + 1.0)
    feats = ["log_pos_info", "log_rot_info", "log_n_vert", "normal_ev_min", "speed"]
    X, y = seg[feats].values, seg.y.values
    groups = seg["id"].values
    k = min(cfg.evaluation.cv_folds, len(np.unique(groups)))
    pred = cross_val_predict(LinearRegression(), X, y, cv=GroupKFold(n_splits=k), groups=groups)
    r2_cv = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    single = {}
    for f in feats:
        p1 = cross_val_predict(LinearRegression(), seg[[f]].values, y, cv=GroupKFold(n_splits=k), groups=groups)
        single[f] = 1 - np.sum((y - p1) ** 2) / np.sum((y - y.mean()) ** 2)
    import statsmodels.api as sm
    ols = sm.OLS(y, sm.add_constant(seg[feats])).fit()
    md.append(f"Yalnız-LIO {cfg.evaluation.drift_segment:g} s'lik {len(seg)} bölüt; bağımlı değişken "
              "log10(birim mesafe başına sürüklenme). Bağımsız değişkenler: taramadaki nokta-düzlem kısıt "
              "bilgisinin en küçük konum/yönelim özdeğeri (log), dikey yüzey öznitelik sayısı (log), düzlem "
              "normali dağılımının en küçük özdeğeri ve hız. Uçuş bazında gruplu "
              f"{k}-katlı çapraz doğrulama.\n")
    md.append(f"**Çok değişkenli model: eğitim R² = {ols.rsquared:.3f}, çapraz doğrulama R² = {r2_cv:.3f}**\n")
    coef = pd.DataFrame({"Değişken": ols.params.index, "Katsayı": ols.params.values.round(4),
                         "p": ols.pvalues.values, "Tek değişkenli CV R²": [""] + [f"{single[f]:.3f}" for f in feats]})
    coef["p"] = coef.p.map(lambda v: f"{v:.2g}")
    md.append(_md_table(coef) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for e, c in ENV_COLORS.items():
        s = seg[seg.env == e]
        if s.empty:
            continue
        axes[0].scatter(s.log_pos_info, s.y, s=10, alpha=0.55, color=c, label=ENV_LABELS[e], edgecolors="none")
    axes[0].set_xlabel("log10(en küçük konum kısıt bilgisi özdeğeri)")
    axes[0].set_ylabel("log10(sürüklenme / mesafe)")
    axes[0].set_title("Sürüklenme – geometrik kısıt zenginliği")
    axes[0].legend(loc="upper right", fontsize=8.5)
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    axes[1].plot(lim, lim, color=INK2, lw=1, ls="--")
    for e, c in ENV_COLORS.items():
        m = seg.env.values == e
        axes[1].scatter(pred[m], y[m], s=10, alpha=0.55, color=c, edgecolors="none", label=ENV_LABELS[e])
    axes[1].set_xlabel("Çapraz doğrulama tahmini")
    axes[1].set_ylabel("Gözlenen")
    axes[1].set_title(f"Model uyumu (CV R² = {r2_cv:.2f})")
    axes[1].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "drift_geometry.png")
    plt.close(fig)
    md.append("![Sürüklenme-geometri](figures/drift_geometry.png)\n")
    return {"H3_cv_r2": float(r2_cv), "H3_train_r2": float(ols.rsquared), "H3_single_r2": single}


def _example_plot(out_dir, fig_dir, md):
    files = sorted((out_dir / "runs").glob("campus_*.npz")) or sorted((out_dir / "runs").glob("*.npz"))
    if not files:
        return
    d = np.load(files[0])
    meta = json.loads(str(d["meta"]))
    t, gt = d["t"], d["g_t"]
    t_on = meta["onset"]
    t_off = t_on + meta["denied"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2, 1.0]})
    for m in ["gnss_ins", "monitor", "hard", "voting", "soft"]:
        axes[0].plot(t, d[f"slow_drift/{m}/err"], color=METHOD_COLORS[m], label=METHOD_LABELS[m],
                     lw=2.4 if m == "soft" else 1.6)
    axes[0].set_ylabel("Konum hatası [m]")
    axes[0].set_title(f"Örnek: yavaş sürüklemeli aldatma — {meta['id']}")
    axes[0].legend(fontsize=8, ncols=2, loc="upper left")
    for m in ["hard", "soft"]:
        axes[1].plot(gt, np.minimum(d[f"slow_drift/{m}/rho"], 5), color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    axes[1].axhline(1.0, color=INK2, lw=1, ls="--")
    axes[1].set_ylabel("Tutarlılık ρ")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[2].semilogy(gt, np.maximum(d["slow_drift/soft/lam"], 1), color=METHOD_COLORS["soft"])
    axes[2].set_ylabel("GNSS kovaryans\nşişirme λ")
    axes[2].set_xlabel("Zaman [s]")
    for ax in axes:
        ax.axvspan(t_on, t_off, color="#e34948", alpha=0.07, lw=0)
    axes[0].annotate("aldatma penceresi", (t_on + 1, 0.04), xycoords=("data", "axes fraction"),
                     color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "example_slow_drift.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ref = d["ref_p"]
    ax.plot(ref[:, 0], ref[:, 1], color=INK2, lw=1.2, label="Referans (RTK)")
    g = d["slow_drift/soft/gnss_pos"]
    ax.plot(g[:, 0], g[:, 1], ".", ms=2, color=METHOD_COLORS["gnss_ins"], label="Aldatılmış GNSS")
    for m in ["monitor", "soft"]:
        p = d[f"slow_drift/{m}/p"]
        ax.plot(p[:, 0], p[:, 1], color=METHOD_COLORS[m], lw=1.6, label=METHOD_LABELS[m])
    ax.set_aspect("equal")
    ax.set_xlabel("Doğu [m]")
    ax.set_ylabel("Kuzey [m]")
    ax.set_title("Üstten görünüm — yavaş sürükleme")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "example_topview.png")
    plt.close(fig)
    md.append("## 7. Örnek uçuş\n")
    md.append("![Örnek zaman serisi](figures/example_slow_drift.png)\n")
    md.append("![Üstten görünüm](figures/example_topview.png)\n")
