"""Hata-durumlu genişletilmiş Kalman filtresi (ESKF) için INS çekirdeği.

Nominal durum : p, v, R, ba, bg   (dünya = yerel ENU, yerçekimi -z)
Hata durumu   : [δp, δv, δθ, δba, δbg]  (δθ yerel/gövde perturbasyonu: R = R̂ Exp(δθ))
İsteğe bağlı olarak sona "klon" durumları (p_c, θ_c) eklenebilir (stokastik klonlama).
"""
import numpy as np

from .geometry import skew, so3_exp

GRAVITY = np.array([0.0, 0.0, -9.81])
N_CORE = 15


class Ins:
    def __init__(self, p, v, R, P0, imu_cfg, ba=None, bg=None):
        self.p = np.array(p, float)
        self.v = np.array(v, float)
        self.R = np.array(R, float)
        self.ba = np.zeros(3) if ba is None else np.array(ba, float)
        self.bg = np.zeros(3) if bg is None else np.array(bg, float)
        self.P = np.array(P0, float)
        self.clone_p = None
        self.clone_R = None
        rate = imu_cfg.rate
        # sürekli zaman gürültü yoğunlukları (biraz muhafazakâr)
        self.q_acc = (imu_cfg.accel_noise * 1.5) ** 2
        self.q_gyro = (imu_cfg.gyro_noise * 1.5) ** 2
        self.q_ba = (imu_cfg.accel_bias_rw * 2.0) ** 2
        self.q_bg = (imu_cfg.gyro_bias_rw * 2.0) ** 2
        self.dt_nom = 1.0 / rate

    @staticmethod
    def default_P0(sig_p=1.0, sig_v=0.2, sig_th_deg=1.0, sig_ba=0.05, sig_bg=0.005):
        return np.diag(np.r_[np.full(3, sig_p ** 2), np.full(3, sig_v ** 2),
                             np.full(3, np.deg2rad(sig_th_deg) ** 2),
                             np.full(3, sig_ba ** 2), np.full(3, sig_bg ** 2)])

    # ------------------------------------------------------------ yayılım
    def propagate(self, acc, gyro, dt):
        R = self.R
        f = acc - self.ba
        w = gyro - self.bg
        a_w = R @ f + GRAVITY
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        dR = so3_exp(w * dt)
        self.R = R @ dR

        F = np.eye(N_CORE)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ skew(f) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = dR.T
        F[6:9, 12:15] = -np.eye(3) * dt
        P = self.P
        Pmm = F @ P[:N_CORE, :N_CORE] @ F.T
        Pmm[3:6, 3:6] += np.eye(3) * self.q_acc * dt
        Pmm[6:9, 6:9] += np.eye(3) * self.q_gyro * dt
        Pmm[9:12, 9:12] += np.eye(3) * self.q_ba * dt
        Pmm[12:15, 12:15] += np.eye(3) * self.q_bg * dt
        if P.shape[0] > N_CORE:
            Pmc = F @ P[:N_CORE, N_CORE:]
            P[:N_CORE, N_CORE:] = Pmc
            P[N_CORE:, :N_CORE] = Pmc.T
        P[:N_CORE, :N_CORE] = Pmm

    # ------------------------------------------------------------ düzeltme
    def inject(self, dx):
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.R = self.R @ so3_exp(dx[6:9])
        self.ba += dx[9:12]
        self.bg += dx[12:15]
        if self.clone_p is not None and len(dx) > N_CORE:
            self.clone_p = self.clone_p + dx[15:18]
            self.clone_R = self.clone_R @ so3_exp(dx[18:21])

    def update(self, H, r, Rm):
        """Standart EKF güncellemesi (Joseph biçimi). r = z - h(x̂)."""
        P = self.P
        PHt = P @ H.T
        S = H @ PHt + Rm
        K = np.linalg.solve(S, PHt.T).T
        dx = K @ r
        IKH = np.eye(P.shape[0]) - K @ H
        self.P = IKH @ P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.inject(dx)
        return dx

    # ------------------------------------------------------------ stokastik klonlama
    def set_clone(self):
        """Mevcut (p, θ) durumunu klonlar; kovaryansı 21x21'e genişletir."""
        Pm = self.P[:N_CORE, :N_CORE]
        J = np.zeros((6, N_CORE))
        J[0:3, 0:3] = np.eye(3)
        J[3:6, 6:9] = np.eye(3)
        P = np.zeros((N_CORE + 6, N_CORE + 6))
        P[:N_CORE, :N_CORE] = Pm
        P[:N_CORE, N_CORE:] = Pm @ J.T
        P[N_CORE:, :N_CORE] = J @ Pm
        P[N_CORE:, N_CORE:] = J @ Pm @ J.T
        self.P = P
        self.clone_p = self.p.copy()
        self.clone_R = self.R.copy()
