"""Komut satırı arayüzü.

    python -m gnss_lio_sim all      [--quick]     # benzetim + değerlendirme + analiz (tam deney)
    python -m gnss_lio_sim simulate [--quick]
    python -m gnss_lio_sim evaluate [--quick]
    python -m gnss_lio_sim analyze  [--quick]
    python -m gnss_lio_sim demo     [--env campus --traj figure8 --attack slow_drift --method soft]
    python -m gnss_lio_sim export-bag [--env campus --attack slow_drift]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from .config import ROOT, load_config

QUICK = {
    "flight": {"trajectories": ["rectangle", "figure8"], "speeds": [5.0], "repetitions": 2,
               "attack_onset": [25.0, 30.0], "gnss_denied_duration": 60.0, "recovery_duration": 15.0},
}


def _setup(args):
    overrides = QUICK if args.quick else None
    cfg = load_config(args.config, overrides)
    profile = "quick" if args.quick else "full"
    out = Path(args.out) if args.out else ROOT / "results" / profile
    out.mkdir(parents=True, exist_ok=True)
    return cfg, overrides, out


def cmd_simulate(args):
    from .experiment import simulate_all
    cfg, ov, out = _setup(args)
    simulate_all(cfg, args.config, ov, out / "flights", args.workers)


def cmd_evaluate(args):
    from .experiment import evaluate_all
    cfg, ov, out = _setup(args)
    evaluate_all(cfg, args.config, ov, out / "flights", out, args.workers, severity=not args.no_severity)


def cmd_analyze(args):
    from .analysis import analyze
    cfg, ov, out = _setup(args)
    analyze(out, cfg)


def cmd_tune(args):
    from .experiment import default_workers
    from .tuning import tune
    args.quick = True  # ayar her zaman geliştirme (hızlı profil) uçuşlarında yapılır
    cfg, ov, out = _setup(args)
    if not any((out / "flights").glob("*.npz")):
        cmd_simulate(args)
    tune(args.config, ov, out / "flights", out, args.workers or default_workers())


def cmd_all(args):
    cmd_simulate(args)
    cmd_evaluate(args)
    cmd_analyze(args)


def _demo_flight(args, cfg):
    from .sim.flight import build_world, flight_id, load_flight, load_scans, simulate_flight
    d = ROOT / "results" / "demo"
    fid = flight_id(args.env, args.traj, args.speed, args.rep)
    if not (d / "scans" / f"{fid}_scans.npz").exists():
        print(f"[demo] {fid} uçuşu benzetiliyor (LiDAR taramaları kaydediliyor)...")
        simulate_flight(cfg, args.env, args.traj, args.speed, args.rep, d, save_scans=True)
    flight = load_flight(d / f"{fid}.npz")
    _, scans = load_scans(d / "scans" / f"{fid}_scans.npz")
    return d, fid, flight, scans


def cmd_demo(args):
    from .fusion import run_fusion
    from .metrics import eval_run
    from .sim.flight import build_world
    from .spoofing import inject
    cfg = load_config(args.config)
    d, fid, flight, scans = _demo_flight(args, cfg)
    meta = flight["meta"]
    print(f"[demo] uçuş {fid}: aldatma başlangıcı {meta['onset']:.1f} s, süre {meta['denied']:.0f} s")
    results = {}
    for m in ["gnss_ins", "monitor", "hard", "voting", "soft"]:
        att = inject(flight["gnss_t"], flight["gnss_pos"], flight["gnss_vel"], meta, args.attack, cfg)
        res = run_fusion(flight, att, m, cfg)
        r = eval_run(flight, att, res, cfg)
        results[m] = (att, res)
        print(f"  {m:9s} RMSE {r['rmse']:6.2f} m  maks {r['max_err']:6.2f} m  "
              f"tespit gecikmesi {r.get('delay', np.nan):6.2f} s  ek hata {r.get('peak_add_err', np.nan):6.2f} m")
    if args.no_3d:
        return
    from .replay3d import replay
    w = build_world(cfg, args.env, args.traj, args.speed, args.rep)
    att, res = results[args.method]
    replay(w["env"], flight, scans, res, att,
           title=f"{fid} — {args.attack} / {args.method}", speed=args.play_speed)


def cmd_export_bag(args):
    from .rosbag_export import export_bag
    from .spoofing import inject
    cfg = load_config(args.config)
    d, fid, flight, scans = _demo_flight(args, cfg)
    att = inject(flight["gnss_t"], flight["gnss_pos"], flight["gnss_vel"], flight["meta"], args.attack, cfg)
    out = export_bag(flight, scans, att, d / "bags" / f"{fid}_{args.attack}", cfg)
    print(f"[export-bag] ROS 2 bag yazıldı: {out}")


def cmd_real_lio(args):
    import json
    from .real.pipeline import evaluate_lio, run_lio, save_lio
    cfg = load_config(args.config or ROOT / "config" / "mars_lvig.yaml")
    out = Path(args.out or ROOT / "results" / "real")
    for seq in args.seq:
        _, ref, recs = run_lio(cfg, seq)
        res, series = evaluate_lio(recs, ref, cfg)
        save_lio(out, seq, recs, res, series, ref)
        print(f"[real-lio] {seq}:", json.dumps(res, indent=1, ensure_ascii=False))


def cmd_real_eval(args):
    from .experiment import evaluate_all
    from .real.pipeline import (build_flights, evaluate_lio, load_seq, reference, rtk_dup_fraction, run_lio,
                                save_lio)
    from .real.report import report
    cfg_path = args.config or ROOT / "config" / "mars_lvig.yaml"
    cfg = load_config(cfg_path)
    out = Path(args.out or ROOT / "results" / "real")
    flights = out / "flights"
    excluded = {}
    for seq in args.seq:
        dup = rtk_dup_fraction(cfg, seq)
        if dup > cfg.real.max_rtk_dup:
            excluded[seq] = dup
            print(f"[real-eval] {seq} dışlandı: RTK referans mesajlarının %{100 * dup:.1f}'i tekrar (bayat)")
    seqs = [s for s in args.seq if s not in excluded]
    # u-blox hata dizisinin kaynağı, yarı-gerçek dizilerden önce işlenmeli
    seqs.sort(key=lambda s: s != cfg.real.gnss_donor)
    for f in flights.glob("*.npz"):  # önceki koşulardan kalan uçuş kayıtları karışmasın
        f.unlink()
    for seq in seqs:
        if not (out / f"{seq}_lio.npz").exists():
            _, ref, recs = run_lio(cfg, seq)
            res, series = evaluate_lio(recs, ref, cfg)
            save_lio(out, seq, recs, res, series, ref)
        build_flights(cfg, seq, out, flights)
    evaluate_all(cfg, str(cfg_path), None, flights, out, workers=args.workers)
    replay = all(reference(load_seq(cfg, s)).get("synthetic_gnss", False) for s in seqs)
    report(out, cfg, seqs, out, excluded=excluded, replay=replay)


def cmd_drift_sweep(args):
    from .drift_sweep import min_rate, run
    from .experiment import default_workers
    real_cfg = str(ROOT / "config" / "mars_lvig.yaml")
    out = Path(args.out or ROOT / "results" / "drift_sweep")
    _, summary = run(_eval_jobs(args.groups), out, args.workers or default_workers(), groups=args.groups)
    print(summary.round(3).to_string(index=False))
    for g in summary.group.unique():
        print(f"[drift-sweep] {g}: 5 m öncesi tespit ≥ %95 için en küçük hız {min_rate(summary, g)} m/s")


def cmd_baselines(args):
    from .baselines import compare, run, tune
    from .experiment import default_workers
    out = Path(args.out or ROOT / "results" / "baselines")
    out.mkdir(parents=True, exist_ok=True)
    workers = args.workers or default_workers()
    if args.action == "tune":
        print(tune(ROOT / "results" / "quick" / "flights", QUICK, out, workers, methods=args.methods))
        return
    if args.action == "run":
        run(_eval_jobs(args.groups), out, workers, methods=args.methods, groups=args.groups)
    print(compare(ROOT, out).round(2).to_string(index=False))


def _eval_jobs(groups=None):
    """Gerçek veri (MARS-LVIG 'real', MUN-FRL 'munfrl' / 'munfrl_rtk'), yarı-gerçek veri ('semireal')
    + tam benzetim uçuşları (grup = ortam)."""
    cfgs = {"real": "mars_lvig", "semireal": "mars_lvig", "munfrl": "mun_frl", "munfrl_rtk": "mun_frl_rtk"}
    jobs = [(str(ROOT / "config" / f"{c}.yaml"), f, g) for g, c in cfgs.items()
            for f in sorted((ROOT / "results" / g / "flights").glob("*.npz"))]
    jobs += [(None, f, None) for f in sorted((ROOT / "results" / "full" / "flights").glob("*.npz"))]
    if groups:  # benzetim uçuşlarının grubu ortam adıdır
        from .sim.flight import load_flight
        jobs = [j for j in jobs if (j[2] or load_flight(j[1])["meta"]["env"]) in groups]
    return jobs


def cmd_roc(args):
    from .experiment import default_workers
    from .roc import run
    out = Path(args.out or ROOT / "results" / "roc")
    _, s = run(_eval_jobs(args.groups), out, args.workers or default_workers(), groups=args.groups)
    print(s.round(3).to_string(index=False))


def cmd_sensitivity(args):
    from .experiment import default_workers
    from .sensitivity import run
    out = Path(args.out or ROOT / "results" / "sensitivity")
    _, s = run(_eval_jobs(args.groups), out, args.workers or default_workers(), groups=args.groups)
    print(s.round(3).to_string(index=False))



def cmd_adaptive(args):
    from .adaptive import run
    from .experiment import default_workers
    out = Path(args.out or ROOT / "results" / "adaptive")
    _, s = run(_eval_jobs(args.groups), out, args.workers or default_workers(), groups=args.groups)
    print(s.round(2).to_string(index=False))

def main(argv=None):
    for s in (sys.stdout, sys.stderr):  # Türkçe Windows'ta dosyaya yönlendirilen çıktı cp1254 olur
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="gnss_lio_sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", default=None, help="ek YAML yapılandırma (varsayılanın üzerine yazar)")
        sp.add_argument("--quick", action="store_true", help="hızlı deneme profili (az uçuş)")
        sp.add_argument("--out", default=None, help="çıktı klasörü")
        sp.add_argument("--workers", type=int, default=None, help="paralel işçi sayısı")
        sp.add_argument("--no-severity", action="store_true", help="şiddet taramasını atla")

    for name, fn in (("simulate", cmd_simulate), ("evaluate", cmd_evaluate),
                     ("analyze", cmd_analyze), ("all", cmd_all), ("tune", cmd_tune)):
        sp = sub.add_parser(name)
        common(sp)
        sp.set_defaults(fn=fn)

    def flight_args(sp):
        sp.add_argument("--config", default=None)
        sp.add_argument("--env", default="campus", choices=["open_field", "campus", "flat_poor"])
        sp.add_argument("--traj", default="figure8", choices=["line", "rectangle", "figure8"])
        sp.add_argument("--speed", type=float, default=5.0)
        sp.add_argument("--rep", type=int, default=0)
        sp.add_argument("--attack", default="slow_drift",
                        choices=["clean", "jump", "ramp", "slow_drift", "noise", "dropout"])

    sp = sub.add_parser("demo", help="tek uçuş + 3B oynatma")
    flight_args(sp)
    sp.add_argument("--method", default="soft", choices=["gnss_ins", "monitor", "hard", "voting", "soft"])
    sp.add_argument("--no-3d", action="store_true")
    sp.add_argument("--play-speed", type=float, default=2.0)
    sp.set_defaults(fn=cmd_demo)
    sp = sub.add_parser("export-bag", help="uçuşu ROS 2 bag olarak dışa aktar")
    flight_args(sp)
    sp.set_defaults(fn=cmd_export_bag)
    sp = sub.add_parser("real-lio", help="MARS-LVIG gerçek verisinde yalnız-LIO taban hattı")
    sp.add_argument("--config", default=None, help="varsayılan: config/mars_lvig.yaml")
    sp.add_argument("--seq", nargs="+", default=["HKairport_GNSS03"], help="dizi adları (bag dosya adı)")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_real_lio)
    sp = sub.add_parser("real-eval", help="MARS-LVIG gerçek verisinde tam aldatma deneyi + rapor")
    sp.add_argument("--config", default=None, help="varsayılan: config/mars_lvig.yaml")
    sp.add_argument("--seq", nargs="+", default=["HKairport_GNSS03"])
    sp.add_argument("--out", default=None)
    sp.add_argument("--workers", type=int, default=None)
    sp.set_defaults(fn=cmd_real_eval)
    sp = sub.add_parser("drift-sweep", help="yavaş sürükleme hızı taraması (tespit sınırı)")
    sp.add_argument("--groups", nargs="+", default=None, help="yalnız bu gruplar (ör. munfrl munfrl_rtk)")
    sp.add_argument("--out", default=None)
    sp.add_argument("--workers", type=int, default=None)
    sp.set_defaults(fn=cmd_drift_sweep)

    sp = sub.add_parser("roc", help="yalnız-tespit ROC/AUC karşılaştırması (önerilen + literatür yöntemleri)")
    sp.add_argument("--groups", nargs="+", default=None, help="yalnız bu gruplar (ör. munfrl munfrl_rtk)")
    sp.add_argument("--out", default=None)
    sp.add_argument("--workers", type=int, default=None)
    sp.set_defaults(fn=cmd_roc)
    sp = sub.add_parser("baselines", help="literatür karşılaştırma yöntemleri (Tanil 2018, Gu 2021, Aljaafreh 2026)")
    sp.add_argument("--groups", nargs="+", default=None, help="yalnız bu gruplar (ör. munfrl munfrl_rtk)")
    sp.add_argument("action", choices=["tune", "run", "compare"])
    sp.add_argument("--methods", nargs="+", default=None, help="yalnız bu yöntemler (ör. alja_lio)")
    sp.add_argument("--out", default=None)
    sp.add_argument("--workers", type=int, default=None)
    sp.set_defaults(fn=cmd_baselines)

    sp = sub.add_parser("adaptive", help="uyarlanabilir (eşik altı) saldırgan testi, en kötü durum")
    sp.add_argument("--groups", nargs="+", default=None, help="yalnız bu gruplar (ör. real munfrl)")
    sp.add_argument("--out", default=None)
    sp.add_argument("--workers", type=int, default=None)
    sp.set_defaults(fn=cmd_adaptive)

    sp = sub.add_parser("sensitivity", help="lio_unhealthy_hold_nognss duyarlılık analizi")
    sp.add_argument("--groups", nargs="+", default=None, help="yalnız bu gruplar (ör. munfrl munfrl_rtk)")
    sp.add_argument("--out", default=None)
    sp.add_argument("--workers", type=int, default=None)
    sp.set_defaults(fn=cmd_sensitivity)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
