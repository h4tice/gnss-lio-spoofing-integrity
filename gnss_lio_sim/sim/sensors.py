"""Algılayıcı modelleri: MEMS IMU, GNSS (konum+hız), RTK referansı ve 3B LiDAR."""
import numpy as np


def _interval_mean(x, idx, step):
    """IMU örneği k: [t_k, t_k+dt] aralığının yamuk kuralıyla ortalaması
    (gerçek IMU'ların entegre eden/örtüşme önleyici süzgeç davranışı)."""
    w = np.ones(step + 1)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    n = len(x)
    out = np.zeros((len(idx), x.shape[1]))
    for j in range(step + 1):
        out += w[j] * x[np.minimum(idx + j, n - 1)]
    return out


def simulate_imu(truth, cfg, rng):
    """Gerçek durum dizisinden (1 kHz) IMU ölçümleri üretir."""
    step = int(round(truth["rate"] / cfg.rate))
    idx = np.arange(0, len(truth["t"]), step)
    n = len(idx)
    w_true = _interval_mean(truth["w_body"], idx, step)
    f_true = _interval_mean(truth["f_body"], idx, step)
    dt = 1.0 / cfg.rate
    bg0 = rng.normal(0, cfg.gyro_bias_init, 3)
    ba0 = rng.normal(0, cfg.accel_bias_init, 3)
    bg = bg0 + np.cumsum(rng.normal(0, cfg.gyro_bias_rw * np.sqrt(dt), (n, 3)), axis=0)
    ba = ba0 + np.cumsum(rng.normal(0, cfg.accel_bias_rw * np.sqrt(dt), (n, 3)), axis=0)
    gyro = w_true + bg + rng.normal(0, cfg.gyro_noise * np.sqrt(cfg.rate), (n, 3))
    acc = f_true + ba + rng.normal(0, cfg.accel_noise * np.sqrt(cfg.rate), (n, 3))
    return {"t": truth["t"][idx], "gyro": gyro, "acc": acc, "bg": bg, "ba": ba}


def simulate_gnss(truth, cfg, rng):
    """Sağlıklı (aldatılmamış) GNSS konum/hız ölçümleri. Aldatma sonradan enjekte edilir."""
    step = int(round(truth["rate"] / cfg.rate))
    idx = np.arange(0, len(truth["t"]), step)
    n = len(idx)
    dt = 1.0 / cfg.rate
    # birinci dereceden Gauss-Markov ilintili hata (çok yol / atmosfer benzeri)
    phi = np.exp(-dt / cfg.gm_tau)
    gm = np.zeros((n, 3))
    gm[0] = rng.normal(0, cfg.gm_sigma, 3)
    qs = cfg.gm_sigma * np.sqrt(1 - phi ** 2)
    for k in range(1, n):
        gm[k] = phi * gm[k - 1] + rng.normal(0, qs, 3)
    gm[:, 2] *= 1.5
    white = rng.normal(0, 1, (n, 3)) * np.array([cfg.sigma_h, cfg.sigma_h, cfg.sigma_v])
    pos = truth["p"][idx] + gm + white
    vel = truth["v"][idx] + rng.normal(0, cfg.vel_sigma, (n, 3))
    return {"t": truth["t"][idx], "pos": pos, "vel": vel}


def simulate_rtk(truth, sigma, rate, rng):
    """Santimetre doğruluklu bağımsız referans (ground truth) yörünge."""
    step = int(round(truth["rate"] / rate))
    idx = np.arange(0, len(truth["t"]), step)
    return {"t": truth["t"][idx],
            "p": truth["p"][idx] + rng.normal(0, sigma, (len(idx), 3)),
            "R": truth["R"][idx], "v": truth["v"][idx]}


class Lidar:
    """Unitree 4D LiDAR L2 benzeri tekrarsız taramalı LiDAR (aşağı bakan montaj).

    Her taramada FOV içinde düzgün dağılımlı rastgele ışınlar (tekrarsız desen),
    menzil gürültüsü ve menzil sınırları uygulanır. Tarama, tarama anındaki
    gövde pozundan alınır (hareket bozulması ihmal edilir).
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.z_lo = np.sin(np.deg2rad(cfg.elev_min_deg))
        self.z_hi = np.sin(np.deg2rad(cfg.elev_max_deg))

    def scan(self, p, R, rng):
        n = self.cfg.points_per_scan
        z = rng.uniform(self.z_lo, self.z_hi, n)  # küre üzerinde düzgün dağılım
        az = rng.uniform(-np.pi, np.pi, n)
        rxy = np.sqrt(1 - z ** 2)
        d_b = np.stack([rxy * np.cos(az), rxy * np.sin(az), z], axis=1)
        d_w = d_b @ R.T
        t_hit = self.env.cast(np.repeat(p[None], n, axis=0), d_w)
        ok = np.isfinite(t_hit) & (t_hit > self.cfg.min_range) & (t_hit < self.cfg.max_range)
        rng_m = t_hit[ok] + rng.normal(0, self.cfg.range_noise, ok.sum())
        return (d_b[ok] * rng_m[:, None]).astype(np.float32)
