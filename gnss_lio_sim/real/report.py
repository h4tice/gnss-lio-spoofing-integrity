"""Gerçek veri (MARS-LVIG) deney raporu: results.csv → report.md + şekiller."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ATTACK_TR = {"jump": "Ani sıçrama", "ramp": "Sabit kayma", "slow_drift": "Yavaş sürükleme",
             "noise": "Gürültü artışı", "dropout": "Tam sinyal kaybı", "clean": "Saldırısız"}
METHOD_TR = {"gnss_ins": "GNSS/INS (LIO yok)", "monitor": "İzleme (dışlama yok)", "hard": "Sert anahtarlama",
             "voting": "Sabit eşikli oylama", "soft": "Sürekli ağırlıklandırma (önerilen)", "lio_only": "Yalnız LIO"}
COLORS = {"hard": "#2a78d6", "voting": "#eb6834", "soft": "#1baf7a", "gnss_ins": "#eda100",
          "monitor": "#e87ba4", "lio_only": "#008300"}
LIMIT_M = 5.0  # yavaş sürükleme/kayma için tespit sınırı: kasıtlı öteleme 5 m'ye ulaşmadan


def _ms(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "—"
    return f"{x.mean():.2f} ± {x.std():.2f}" if len(x) > 1 else f"{x.mean():.2f}"


def _md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def report(out_dir, cfg, seqs, lio_dir, excluded=None, replay=False, log=print):
    out = Path(out_dir)
    df = pd.read_csv(out / "results.csv")
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    md = [f"# Gerçek veri doğrulaması — {cfg.real.get('title', 'MARS-LVIG (DJI M300 + Livox Avia + ZED-F9P)')}", ""]
    md.append(f"Diziler: {', '.join(seqs)} · uçuş kaydı (saldırı başlangıcı) sayısı: {df['id'].nunique()} · "
              f"koşu sayısı: {len(df)}")
    md.append("")
    if replay:
        md.append(f"Deney GNSS'i (**yarı-gerçek**): bu dizilerde u-blox alıcısı kaydedilmemiştir. GNSS ölçümü, "
                  f"bağımsız DJI M300 RTK yörüngesine {cfg.real.gnss_donor} uçuşunda ölçülmüş gerçek u-blox "
                  "hata dizisi (konum ve hız) eklenerek oluşturulmuştur; LiDAR ve IMU tamamen gerçektir. "
                  "Referans: aynı RTK çözümü.")
        md.append("")
    md.append(cfg.real.get("gnss_desc", "Deney GNSS'i: ZED-F9P alıcısının kendi (RTK'sız) konum/hız çözümü, 5 Hz. "
              "Referans: bağımsız DJI M300 RTK (sabit çözüm).") +
              " Saldırılar benzetimdekiyle aynı yöntemle (kayıt-üzerinde-oynatma) GNSS "
              "ölçümlerine yazılımla eklenmiştir. Tespit eşikleri **benzetimin geliştirme kümesinden aynen "
              if not replay else
              "Saldırılar benzetimdekiyle aynı yöntemle GNSS ölçümlerine yazılımla eklenmiştir. Tespit eşikleri "
              "**benzetimin geliştirme kümesinden aynen "
              "alınmış**, gerçek veride yeniden ayarlanmamıştır; yalnız fiziksel modeller (GNSS hata modeli, LIO "
              "göreli kovaryans ölçeği) veriden kalibre edilmiştir. Testler ve hata ölçütleri yatay düzlemdedir.")
    md.append("")
    for s, dup in (excluded or {}).items():
        md.append(f"Dışlanan dizi: **{s}** — DJI RTK referans mesajlarının %{100 * dup:.1f}'i aynı konumun bayat "
                  f"tekrarıdır (ölçüt: > %{100 * cfg.real.max_rtk_dup:.0f}); referans zamanlaması güvenilir değildir.")
        md.append("")

    # ---- LIO taban hattı
    md += ["## 1. Yalnız-LIO taban hattı (GNSS kullanılmadan)", ""]
    rows = []
    for s in seqs:
        p = Path(lio_dir) / f"{s}_lio.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"Dizi": s, "Yol [m]": f"{r['path_length']:.0f}", "ATE hizalı [m]": f"{r['ate_global_aligned']:.2f}",
                         "Uç hata (ilk 60 s hizalı) [m]": f"{r['final_err_start_aligned']:.2f}",
                         "60 s sürüklenme ort/en kötü [%]": f"{r['drift60_pct_mean']:.2f} / {r['drift60_pct_max']:.2f}",
                         "RPE 1 s [m]": f"{r['rpe_1s']:.2f}",
                         "İşlem ort/p95 [ms]": f"{r['proc_ms_mean']:.0f} / {r['proc_ms_p95']:.0f}"})
    if rows:
        md += [_md(pd.DataFrame(rows)), ""]
        for s in seqs:
            if (fig_dir / f"{s}_lio.png").exists():
                md += [f"![{s} LIO](figures/{s}_lio.png)", ""]

    att = df[(df.severity == 1.0) & (df.method != "lio_only")]
    # ---- yanlış alarm
    md += ["## 2. Yanlış alarm (saldırısız ve saldırı öncesi sağlıklı çağlar)", ""]
    fa = []
    for m in ("gnss_ins", "monitor", "hard", "voting", "soft"):
        d = att[att.method == m]
        fa.append({"Yöntem": METHOD_TR[m], "Görev bölümünde YA oranı": f"{100 * d.far.mean():.2f}%",
                   "Tüm uçuşta YA oranı (görev sonu yerinde dönüş dahil)": f"{100 * d.far_full.mean():.2f}%",
                   "En az bir YA içeren temiz koşu": f"{100 * d[d.attack == 'clean'].any_false_alarm.mean():.0f}%"})
    md += [_md(pd.DataFrame(fa)), ""]
    md.append("Görev sonu: drone tarama bacaklarını bitirince yerinde ~190°/s dönmektedir. Bu sırada GNSS "
              "anteninin kaldıraç kolu (modellenmedi) hız testini tetikler; bu bölüm ana ölçütlerin dışında "
              "tutulmuş, etkisi ayrı sütunda gösterilmiştir.")
    md.append("")

    # ---- tespit
    md += ["## 3. Aldatma tespiti", ""]
    det = []
    for a in ("jump", "ramp", "slow_drift", "noise", "dropout"):
        for m in ("hard", "voting", "soft"):
            d = att[(att.attack == a) & (att.method == m)]
            if d.empty:
                continue
            early = (d.detected & (d.offset_at_det < LIMIT_M)).mean() if a in ("ramp", "slow_drift") else np.nan
            det.append({"Saldırı": ATTACK_TR[a], "Yöntem": METHOD_TR[m],
                        "Tespit oranı": f"{100 * d.detected.mean():.0f}%",
                        "5 m öncesi": "—" if np.isnan(early) else f"{100 * early:.0f}%",
                        "Gecikme [s]": _ms(d.delay), "Tespitte öteleme [m]": _ms(d.offset_at_det)})
    md += [_md(pd.DataFrame(det)), ""]
    if "env" in att.columns and att.env.nunique() > 1:
        md += ["Önerilen yöntemin ortama ve uçuşa göre dökümü:", ""]
        env_rows = []
        for key, g in att[att.method == "soft"].groupby(["env", "traj"]):
            sd = g[g.attack == "slow_drift"]
            env_rows.append({"Ortam": key[0], "Uçuş": key[1], "Başlangıç sayısı": g.id.nunique(),
                             "YA (görev)": f"{100 * g.far.mean():.2f}%",
                             "Sıçrama gecikmesi [s]": _ms(g[g.attack == "jump"].delay),
                             "Yavaş sürükleme 5 m öncesi": f"{100 * (sd.detected & (sd.offset_at_det < LIMIT_M)).mean():.0f}%",
                             "GNSS'siz 60 s sürüklenme [%]": _ms(g[g.attack == "dropout"].drift_pct)})
        md += [_md(pd.DataFrame(env_rows)), ""]

    # ---- geçiş
    md += ["## 4. Sarsımsız geçiş (H2) ve GNSS'siz seyir", ""]
    tr = []
    for a in ("jump", "ramp", "slow_drift", "noise", "dropout"):
        for m in ("gnss_ins", "monitor", "hard", "voting", "soft"):
            d = att[(att.attack == a) & (att.method == m)]
            if d.empty:
                continue
            tr.append({"Saldırı": ATTACK_TR[a], "Yöntem": METHOD_TR[m], "Adım süreksizliği [m]": _ms(d.jump_post),
                       "Ek hata sıçraması [m]": _ms(d.peak_add_err), "60 s sürüklenme [%]": _ms(d.drift_pct),
                       "RMSE saldırı sonrası [m]": _ms(d.rmse_post)})
    md += [_md(pd.DataFrame(tr)), ""]
    h = att[att.attack.isin(["jump", "ramp", "slow_drift", "noise"]) & att.method.isin(["hard", "soft"])]
    pv = h.pivot_table(index=["id", "attack"], columns="method", values="jump_post")
    red = 100 * (1 - pv["soft"].mean() / pv["hard"].mean())
    from scipy.stats import wilcoxon
    try:
        p = wilcoxon(pv["soft"], pv["hard"]).pvalue
    except ValueError:
        p = np.nan
    per = {ATTACK_TR[a]: 100 * (1 - g["soft"].mean() / g["hard"].mean())
           for a, g in pv.groupby(level="attack")}
    md.append(f"Sürekli ağırlıklandırmanın sert anahtarlamaya göre geçiş anı yörünge sıçramasında ortalama "
              f"azalma: **{red:.1f}%** (eşleştirilmiş Wilcoxon p = {p:.2g}, n = {len(pv)}). Saldırı türüne göre: "
              + ", ".join(f"{k} {v:.0f}%" for k, v in per.items()) + ".")
    md.append("")
    dd = att[(att.attack == "dropout")]
    md.append("Tam sinyal kaybında 60 s GNSS'siz sürüklenme (kat edilen mesafeye oranla): " + ", ".join(
        f"{METHOD_TR[m]} {_ms(dd[dd.method == m].drift_pct)}%" for m in ("gnss_ins", "hard", "soft")) + ".")
    md.append("")

    # ---- özet
    soft = att[att.method == "soft"]
    sd = soft[soft.attack == "slow_drift"]
    summary = [
        ("H1 — YA oranı < %2 (görev bölümü)", f"{100 * soft.far.mean():.2f}%", soft.far.mean() < 0.02),
        ("H1 — ani sıçramada tespit < 3 s", f"{soft[soft.attack == 'jump'].delay.mean():.2f} s",
         soft[soft.attack == "jump"].delay.mean() < 3),
        ("H1 — yavaş sürükleme 5 m öncesi tespit", f"{100 * (sd.detected & (sd.offset_at_det < LIMIT_M)).mean():.0f}%",
         (sd.detected & (sd.offset_at_det < LIMIT_M)).mean() >= 0.95),
        ("H2 — geçiş sıçramasında ≥ %50 azalma", f"{red:.1f}%", red >= 50),
        ("Hedef 4 — GNSS'siz 60 s sürüklenme < %2", f"{dd[dd.method == 'soft'].drift_pct.mean():.2f}%",
         dd[dd.method == "soft"].drift_pct.mean() < 2),
    ]
    head = ["## Özet", "", "| Ölçüt | Gerçek veri | Durum |", "|---|---|---|"]
    head += [f"| {a} | {b} | {'✅ sağlandı' if c else '❌ sağlanmadı'} |" for a, b, c in summary]
    k = md.index("## 1. Yalnız-LIO taban hattı (GNSS kullanılmadan)")
    md = md[:k] + head + [""] + md[k:]

    _example_plot(out, fig_dir, md)
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[real-report] {out / 'report.md'}")
    for a, b, c in summary:
        log(f"  {a}: {b} {'✓' if c else '✗'}")


def _example_plot(out, fig_dir, md):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = sorted((out / "runs").glob("*.npz"))
    if not runs:
        return
    z = np.load(runs[0])
    meta = json.loads(str(z["meta"]))
    t, gt = z["t"], z["g_t"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    for m in ("monitor", "hard", "soft"):
        ax[0].plot(t, z[f"slow_drift/{m}/err"], color=COLORS[m], lw=1.4, label=METHOD_TR[m])
    for a in ax:
        a.axvspan(meta["onset"], meta["onset"] + meta["denied"], color="#999999", alpha=0.15)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Yatay konum hatası [m]")
    ax[0].set_title(f"{meta['id']} — yavaş sürüklemeli aldatma (gerçek veri)")
    ax[0].legend(fontsize=8)
    for k, c, lab in (("rA", "#2a78d6", "A: pencereli χ²"), ("rB", "#eb6834", "B: CUSUM"),
                      ("rC", "#e87ba4", "C: hız χ²"), ("rD", "#008300", "D: LIO–GNSS yer değiştirme")):
        ax[1].plot(gt, z[f"slow_drift/soft/{k}"], color=c, lw=1, label=lab)
    ax[1].axhline(1, color="#333333", ls="--", lw=1)
    ax[1].set_ylim(0, 3)
    ax[1].set_ylabel("Test / eşik (ρ)")
    ax[1].legend(fontsize=8, ncol=4)
    ax[2].semilogy(gt, np.maximum(z["slow_drift/soft/lam"], 1), color=COLORS["soft"], lw=1.4)
    ax[2].set_ylabel("GNSS şişirme λ")
    ax[2].set_xlabel("Zaman [s]")
    fig.tight_layout()
    fig.savefig(fig_dir / "example_slow_drift.png", dpi=130)
    plt.close(fig)
    md += ["## 5. Örnek uçuş", "", "![Örnek](figures/example_slow_drift.png)", ""]
