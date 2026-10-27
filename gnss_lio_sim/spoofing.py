"""GNSS karıştırma/aldatma enjeksiyonu (yalnızca yazılım düzeyinde; RF yayını yok).

Senaryolar (önerideki bağımsız değişkenler):
  jump        : ani sıçrama   — konuma ani öteleme
  ramp        : sabit kayma   — sabit hızla konum kaydırma (hız ölçümü de tutarlı aldatılır)
  slow_drift  : yavaş sürükleme — konum-izleyen aldatıcı; tespiti en güç senaryo
  noise       : gürültü artışı (karıştırma benzeri bozulma)
  dropout     : tam sinyal kaybı
Senaryolar sözde-rastgele tohumlarla birebir tekrarlanabilir.
"""
import zlib

import numpy as np

ATTACK_TYPES = ["jump", "ramp", "slow_drift", "noise", "dropout"]


def inject(gnss_t, gnss_pos, gnss_vel, meta, attack, cfg, scale=1.0):
    """Sağlıklı GNSS dizisine saldırı uygular.

    Dönüş: dict(t, pos, vel, valid, offset, window=(t_on, t_off))
    `offset`: her ölçüme eklenen kasıtlı konum ötelemesi (değerlendirme için).
    """
    pos = gnss_pos.copy()
    vel = gnss_vel.copy()
    valid = np.ones(len(gnss_t), bool)
    offset = np.zeros_like(pos)
    t_on = meta["onset"]
    t_off = t_on + meta["denied"]
    if attack in (None, "none", "clean"):
        return {"t": gnss_t, "pos": pos, "vel": vel, "valid": valid, "offset": offset,
                "window": (np.inf, np.inf)}
    d = np.asarray(meta["attack_dir"])
    m = (gnss_t >= t_on) & (gnss_t < t_off)
    tau = gnss_t[m] - t_on
    acfg = cfg.attacks[attack]
    seed = (meta["seed"] + zlib.crc32(attack.encode())) % (2 ** 31)
    rng = np.random.default_rng(seed)
    if attack == "jump":
        offset[m] = acfg.magnitude * scale * d
    elif attack in ("ramp", "slow_drift"):
        rate = acfg.rate * scale
        offset[m] = (rate * tau)[:, None] * d
        vel[m] += rate * d
    elif attack == "noise":
        f = acfg.factor * scale
        # gürültü tabanı: saldırıya özgü değer verilmişse o, yoksa GNSS beyaz gürültüsü
        sh, sv = acfg.get("sigma_h", cfg.gnss.sigma_h), acfg.get("sigma_v", cfg.gnss.sigma_v)
        sig = np.array([sh, sh, sv])
        offset[m] = rng.normal(0, 1, (m.sum(), 3)) * sig * np.sqrt(f ** 2 - 1)
        vel[m] += rng.normal(0, cfg.gnss.vel_sigma * np.sqrt(f ** 2 - 1), (m.sum(), 3))
    elif attack == "dropout":
        valid[m] = False
    else:
        raise ValueError(f"bilinmeyen saldırı: {attack}")
    pos += offset
    return {"t": gnss_t, "pos": pos, "vel": vel, "valid": valid, "offset": offset,
            "window": (t_on, t_off)}


class AdaptiveAttacker:
    """Uyarlanabilir (beyaz kutu, kapalı çevrim) aldatıcı: hedef yöntemin normalize karar istatistiğini
    (ρ; 1 = alarm eşiği) her çağda gözler ve ötelemeyi ρ'yu `margin` altında tutacak hızla büyütür:

        r_k = r_max · clip((margin − ρ_{k−1}) / margin, −1, 1),   o_k = o_{k−1} + r_k · Δt · d

    Hız ölçümü öteleme hızıyla tutarlı aldatılır (vel += r_k · d). Saldırgan dedektörün yapısını,
    eşiklerini ve anlık durumunu bilir (Kerckhoffs ilkesi; en güçlü saldırgan varsayımı)."""

    def __init__(self, att, d, margin, r_max, dt):
        self.att, self.d, self.margin, self.r_max, self.dt = att, np.asarray(d, float), margin, r_max, dt
        self.clean_pos, self.clean_vel = att["pos"].copy(), att["vel"].copy()
        self.o = np.zeros(3)
        self.rho = 0.0

    def apply(self, gi, tk):
        t_on, t_off = self.att["window"]
        if not (t_on <= tk < t_off):
            return
        r = self.r_max * float(np.clip((self.margin - self.rho) / self.margin, -1.0, 1.0))
        # öteleme yalnız saldırı yönünde: geri çekilme sıfırda durur (sıfırdan geçip ters yönde büyümez)
        s = max(float(self.o @ self.d) + r * self.dt, 0.0)
        r = (s - float(self.o @ self.d)) / self.dt
        self.o = s * self.d
        self.att["pos"][gi] = self.clean_pos[gi] + self.o
        self.att["vel"][gi] = self.clean_vel[gi] + r * self.d
        self.att["offset"][gi] = self.o

    def observe(self, info):
        rho = info.get("rho", 0.0)
        self.rho = float(rho) if np.isfinite(rho) else 1e9


def inject_adaptive(gnss_t, gnss_pos, gnss_vel, meta, cfg, margin, r_max):
    """Saldırı penceresinde uyarlanabilir aldatıcı (öteleme füzyon sırasında üretilir, bkz. fusion)."""
    att = inject(gnss_t, gnss_pos, gnss_vel, meta, "clean", cfg)
    att["window"] = (meta["onset"], meta["onset"] + meta["denied"])
    d = np.asarray(meta["attack_dir"], float)
    att["adaptive"] = AdaptiveAttacker(att, d, margin, r_max, 1.0 / cfg.gnss.rate)
    return att
