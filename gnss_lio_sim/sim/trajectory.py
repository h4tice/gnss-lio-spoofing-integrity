"""Önceden tanımlı uçuş yörüngeleri (düz hat, dikdörtgen tarama, sekiz çizme).

Konum, hız ve ivme yay-uzunluğu ile parametrelenmiş periyodik kübik spline'dan
analitik olarak; yönelim ise dönerkanatlının diferansiyel düzlük özelliğinden
(itki ekseni = a + g) türetilir. Böylece IMU ölçümleri fiziksel olarak tutarlıdır.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

GRAVITY = 9.81


def _shape(name):
    """Kapalı yol şekli (yoğun örneklenmiş xy)."""
    if name == "line":
        # Düz hat: 100 m'lik doğrultuda gidiş-dönüş (dar dönüşlü yarış pisti)
        L, r = 100.0, 8.0
        pts = []
        for x in np.linspace(-L / 2, L / 2, 200):
            pts.append((x, -r))
        for a in np.linspace(-np.pi / 2, np.pi / 2, 40)[1:-1]:
            pts.append((L / 2 + r * np.cos(a), r * np.sin(a)))
        for x in np.linspace(L / 2, -L / 2, 200):
            pts.append((x, r))
        for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 40)[1:-1]:
            pts.append((-L / 2 + r * np.cos(a), r * np.sin(a)))
        return np.array(pts)
    if name == "rectangle":
        W, H, r = 90.0, 60.0, 8.0
        pts = []
        corners = [(W / 2 - r, H / 2 - r, 0.0), (-W / 2 + r, H / 2 - r, np.pi / 2),
                   (-W / 2 + r, -H / 2 + r, np.pi), (W / 2 - r, -H / 2 + r, 1.5 * np.pi)]
        for cx, cy, a0 in corners:
            for a in np.linspace(a0, a0 + np.pi / 2, 30):
                pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        pts = np.array(pts)
        # kenarları yoğunlaştır
        dense = []
        for i in range(len(pts)):
            p0, p1 = pts[i], pts[(i + 1) % len(pts)]
            n = max(2, int(np.linalg.norm(p1 - p0) / 1.0))
            for s in np.linspace(0, 1, n, endpoint=False):
                dense.append(p0 + s * (p1 - p0))
        return np.array(dense)
    if name == "figure8":
        u = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        A, B = 50.0, 30.0
        return np.stack([A * np.sin(u), B * np.sin(u) * np.cos(u) * 2.0], axis=1)
    raise ValueError(f"bilinmeyen yörünge: {name}")


def _resample_closed(xy, step):
    closed = np.vstack([xy, xy[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    u = np.linspace(0, s[-1], int(round(s[-1] / step)), endpoint=False)  # tam düzgün aralık
    return np.stack([np.interp(u, s, closed[:, 0]), np.interp(u, s, closed[:, 1])], axis=1)


def _smooth_closed(xy, step=0.5, sigma_m=4.0):
    """Düzgün yeniden örnekleme + dairesel Gauss yumuşatma: eğriliği sürekli
    (klotoid benzeri geçişli) kapalı yol; spline salınımlarını ve sahte açısal
    hız sıçramalarını önler."""
    from scipy.ndimage import gaussian_filter1d
    xy = _resample_closed(xy, step)
    xy = gaussian_filter1d(xy, sigma_m / step, axis=0, mode="wrap")
    return _resample_closed(xy, step)


class Trajectory:
    """Zamanın fonksiyonu olarak p, v, a, R, ω (gövde) üretir."""

    def __init__(self, name, speed, altitude, duration, rng, jitter=0.3,
                 hover=2.0, ramp=5.0):
        self.name, self.speed, self.duration = name, speed, duration
        self.hover, self.ramp = hover, ramp
        xy = _smooth_closed(_shape(name))
        closed = np.vstack([xy, xy[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        self.length = s[-1]
        self.spline = CubicSpline(s, closed, bc_type="periodic")
        self.altitude = altitude
        # Yumuşak rastgele sapma (rüzgâr/denetleyici izleme hatası benzeri)
        self.jit_a = rng.uniform(0.3, 1.0, (3, 3)) * jitter
        self.jit_a[2] *= 0.5
        self.jit_f = rng.uniform(0.02, 0.12, (3, 3)) * 2 * np.pi
        self.jit_p = rng.uniform(0, 2 * np.pi, (3, 3))
        self.alt_amp = rng.uniform(0.5, 1.2)
        self.s0 = rng.uniform(0, self.length)  # yol üzerindeki başlangıç noktası

    # -- hız profili: bekleme -> yumuşak hızlanma -> sabit hız
    def _speed_profile(self, t):
        t = np.asarray(t, dtype=float)
        tau = np.clip((t - self.hover) / self.ramp, 0.0, 1.0)
        v = self.speed * (tau - np.sin(2 * np.pi * tau) / (2 * np.pi))
        dv = np.where((t > self.hover) & (t < self.hover + self.ramp),
                      self.speed / self.ramp * (1 - np.cos(2 * np.pi * tau)), 0.0)
        # s(t): hız profilinin integrali (kapalı biçim)
        ta = np.clip(t - self.hover, 0.0, self.ramp)
        s_ramp = self.speed * (ta ** 2 / (2 * self.ramp)
                               + self.ramp / (4 * np.pi ** 2) * (np.cos(2 * np.pi * ta / self.ramp) - 1))
        s = s_ramp + self.speed * np.clip(t - self.hover - self.ramp, 0.0, None)
        return s, v, dv

    def _jitter(self, t):
        t = np.asarray(t)[:, None]
        p = np.zeros((t.shape[0], 3))
        v = np.zeros_like(p)
        a = np.zeros_like(p)
        # bekleme sırasında sapma yok, hızlanmayla birlikte yumuşakça devreye girer
        # (zarf e(t) = 3u²-2u³; çarpımın türevleri tam olarak hesaplanır)
        T = self.ramp * 2
        u = np.clip((t[:, 0] - self.hover) / T, 0, 1)
        inside = (u > 0) & (u < 1)
        e = u ** 2 * (3 - 2 * u)
        de = np.where(inside, 6 * u * (1 - u) / T, 0.0)
        dde = np.where(inside, (6 - 12 * u) / T ** 2, 0.0)
        for ax in range(3):
            A, f, ph = self.jit_a[ax], self.jit_f[ax], self.jit_p[ax]
            p[:, ax] = np.sum(A * np.sin(f * t + ph) - A * np.sin(ph), axis=1)
            v[:, ax] = np.sum(A * f * np.cos(f * t + ph), axis=1)
            a[:, ax] = np.sum(-A * f ** 2 * np.sin(f * t + ph), axis=1)
        e, de, dde = e[:, None], de[:, None], dde[:, None]
        return e * p, de * p + e * v, dde * p + 2 * de * v + e * a

    def sample(self, t):
        """t (N,) -> dict(p, v, a, R, w_body, f_body)."""
        t = np.asarray(t, dtype=float)
        s, sd, sdd = self._speed_profile(t)
        u = (self.s0 + s) % self.length
        c = self.spline(u)
        c1 = self.spline(u, 1)
        c2 = self.spline(u, 2)
        p = np.zeros((len(t), 3))
        v = np.zeros_like(p)
        a = np.zeros_like(p)
        p[:, :2] = c
        v[:, :2] = c1 * sd[:, None]
        a[:, :2] = c2 * sd[:, None] ** 2 + c1 * sdd[:, None]
        # irtifa: yol boyunca hafif dalgalanma
        w_alt = 2 * np.pi / self.length * 2
        p[:, 2] = self.altitude + self.alt_amp * (np.sin(w_alt * s) )
        v[:, 2] = self.alt_amp * w_alt * np.cos(w_alt * s) * sd
        a[:, 2] = self.alt_amp * (-(w_alt ** 2) * np.sin(w_alt * s) * sd ** 2
                                  + w_alt * np.cos(w_alt * s) * sdd)
        jp, jv, ja = self._jitter(t)
        p, v, a = p + jp, v + jv, a + ja
        yaw = np.unwrap(np.arctan2(c1[:, 1], c1[:, 0]))
        R = self._flat_attitude(a, yaw)
        return {"t": t, "p": p, "v": v, "a": a, "R": R}

    @staticmethod
    def _flat_attitude(a, yaw):
        thrust = a + np.array([0, 0, GRAVITY])
        zb = thrust / np.linalg.norm(thrust, axis=1, keepdims=True)
        xc = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb, axis=1, keepdims=True)
        xb = np.cross(yb, zb)
        return np.stack([xb, yb, zb], axis=2)

    def generate(self, rate=1000.0):
        """Yüksek hızlı gerçek durum dizisi + gövde açısal hızı ve özgül kuvvet."""
        n = int(round(self.duration * rate)) + 1
        t = np.arange(n) / rate
        st = self.sample(t)
        R = st["R"]
        rot = Rotation.from_matrix(R)
        # gövde açısal hızı: merkezi fark, ω = Log(R_{k-1}^T R_{k+1}) / (2 dt)
        rel = (rot[:-2].inv() * rot[2:]).as_rotvec() * rate / 2.0
        w = np.vstack([rel[:1], rel, rel[-1:]])
        g = np.array([0, 0, GRAVITY])
        f = np.einsum("nji,nj->ni", R, st["a"] + g)  # R^T (a + g)
        st.update(w_body=w, f_body=f, rate=rate)
        return st

    def path_xy(self, step=1.0):
        u = np.arange(0, self.length, step)
        return self.spline(u)
