"""GNSS bütünlük denetimi: ardışık tutarlılık testleri ve geçiş politikaları.

Testler (öneri Aşama 2; her biri eşiğine oranlanmış istatistik üretir, ρ ≥ 1 ⇒ alarm):
  A  pencereli ki-kare testi   — füzyon filtresinin GNSS konum yeniliği NIS toplamı ~ χ²(3N)
  B  CUSUM ardışık testi        — (konum+hız) NIS/dof üzerinde tek yönlü CUSUM
  C  hız yeniliği ki-kare testi — B planı: hız-yeniliği bileşeni
  D  LIO–GNSS yer değiştirme tutarlılığı — son T saniyede GNSS'in gösterdiği yer
     değiştirme ile LIO'nun ölçtüğü yer değiştirme farkı ~ χ²(3). Birden çok T verilirse
     (disp_windows) her biri ayrı sınanır, α pencereler arasında bölünür (Bonferroni): sabit
     hızlı sürükleme T'lik pencerede en çok hız·T fark yaratır, uzun pencere bunu büyütür. Füzyon filtresinden
     bağımsız olduğu için aldatmanın kestirime emilmesinden etkilenmez; yavaş
     sürüklemeli aldatmaya duyarlıdır. Kovaryansı GNSS beyaz gürültüsü, Gauss-Markov
     ilintili hatası ve LIO göreli kovaryanslarından analitik olarak hesaplanır.

Geçiş politikaları (öneri Aşama 3):
  hard    : sert anahtarlama   — herhangi bir test alarmında GNSS tamamen dışlanır
  voting  : sabit eşikli oylama — 4 testten ≥2'si alarmda ise dışlanır
  soft    : önerilen yöntem — testlerin nominal düzey üstü aşımına bağlı sürekli kovaryans
            şişirme; alarmda fiilen dışlama; şüphe azalınca üstel (sarsımsız) geri kabul
  monitor : yalnızca izleme (GNSS her zaman tam ağırlıkta; ROC analizi için)
"""
from collections import deque

import numpy as np
from scipy.stats import chi2

TESTS = ("rA", "rB", "rC", "rD")


class IntegrityMonitor:
    def __init__(self, dcfg, tcfg, gcfg, policy, use_velocity=True, use_disp=True):
        self.d = dcfg
        self.t = tcfg
        self.policy = policy
        self.use_vel = use_velocity
        self.use_disp = use_disp
        self.win_p = deque(maxlen=dcfg.window)
        self.win_v = deque(maxlen=dcfg.vel_window)
        # yatay bütünlük kipi: testler yalnız yatay bileşenlerle (düşey ayrı izlenir; HPL/VPL gibi)
        self.horiz = bool(dcfg.get("horizontal_only", False))
        self.k = 2 if self.horiz else 3
        k = self.k
        self.thr_p = {n: chi2.ppf(1 - dcfg.chi2_alpha, k * n) for n in range(1, dcfg.window + 1)}
        self.thr_v = {n: chi2.ppf(1 - dcfg.vel_alpha, k * n) for n in range(1, dcfg.vel_window + 1)}
        self.g = 0.0
        self.g_cap = 2.0 * dcfg.cusum_h
        # yer değiştirme testi
        rate = gcfg.rate
        self.n_avg = max(1, int(round(dcfg.disp_avg * rate)))
        windows = list(dcfg.get("disp_windows", [dcfg.disp_window]))
        self.n_lags = [max(1, int(round(w * rate))) for w in windows]
        self.buf = deque(maxlen=max(self.n_lags) + self.n_avg)
        self.thr_d = chi2.ppf(1 - dcfg.disp_alpha / len(windows), k)
        wh = np.diag([gcfg.sigma_h ** 2, gcfg.sigma_h ** 2, gcfg.sigma_v ** 2])
        # Gauss-Markov ilintili hata; düşey eksen ayrı verilmemişse yatayın 1,5 katı std, aynı τ
        s_v = gcfg.get("gm_sigma_v", 1.5 * gcfg.gm_sigma)
        tau_v = gcfg.get("gm_tau_v", gcfg.gm_tau)
        var_gm = np.array([gcfg.gm_sigma ** 2, gcfg.gm_sigma ** 2, s_v ** 2])
        tau = np.array([gcfg.gm_tau, gcfg.gm_tau, tau_v])
        # iki ortalama arasındaki farkın GNSS kaynaklı kovaryansı (pencere başına)
        self.S_gnss = [2.0 * wh / self.n_avg + np.diag(2.0 * var_gm * (1 - np.exp(-w / tau))) for w in windows]
        table = gcfg.get("disp_std_table")
        if table:
            # gerçek alıcı: W saniyelik yatay hata değişiminin std'si veriden ölçülmüş tablo [[W, std], ...]
            # (alıcının iç süzgeci hatayı Gauss-Markov'dan farklı biçimlendirir); düşey eksen modelde kalır
            tw, ts = np.array(table, float).T
            for S, w in zip(self.S_gnss, windows):
                S[0, 0] = S[1, 1] = 2.0 * gcfg.sigma_h ** 2 / self.n_avg + np.interp(w, tw, ts) ** 2
        self.rD = 0.0
        self.disp_available = False
        self.above = 0
        self.grace_until = -np.inf
        self.excluded_since = None
        self.n_watchdog = 0
        self.excluded = False
        self.consistent_since = None
        self.log_lambda = 0.0
        self.last_valid_t = None

    def _watchdog(self, t):
        """GNSS uzun süredir dışlıysa ve LIO ile bağımsız doğrulama mümkün değilse (dejenere
        geometri), aldatmaya dair bağımsız kanıt yoktur: testler sıfırlanır ve GNSS yeniden
        kabul edilir (sürekli yöntemde kademeli). Yapısal olarak fakir ortamın bilinen sınırı."""
        if not self.excluded:
            self.excluded_since = None
            return
        if self.excluded_since is None:
            self.excluded_since = t
        if t - self.excluded_since > self.d.watchdog_time and not self.disp_available:
            self.excluded = False
            self.excluded_since = None
            self.g = 0.0
            self.above = 0
            self.win_p.clear()
            self.win_v.clear()
            self.buf.clear()
            self.rD = 0.0
            self.log_lambda = min(self.log_lambda, self.t.soft_gain)
            self.grace_until = t + self.d.readmit_time
            self.n_watchdog += 1

    def stats(self):
        n = len(self.win_p)
        rA = sum(self.win_p) / self.thr_p[n] if n else 0.0
        rB = self.g / self.d.cusum_h
        m = len(self.win_v)
        rC = sum(self.win_v) / self.thr_v[m] if (m and self.use_vel) else 0.0
        return rA, rB, rC, self.rD

    def _disp_test(self, gnss_pos, D, C, counts):
        """r = GNSS − LIO-ölü-hesap; iki n_avg'lik ortalama arasındaki fark n_lag çağ arayla."""
        self.buf.append((gnss_pos - D, C, counts))
        self.disp_available = False
        r = 0.0
        new = [self.buf[-1 - i] for i in range(min(self.n_avg, len(self.buf)))]
        for n_lag, S_g in zip(self.n_lags, self.S_gnss):
            i0 = len(self.buf) - (n_lag + self.n_avg)
            if i0 < 0:
                continue
            n0, d0 = self.buf[i0][2]
            n1, d1 = self.buf[-1][2]
            if n1 > n0 and (d1 - d0) / (n1 - n0) > self.d.disp_max_deg_frac:
                continue  # geometri LIO ile doğrulamaya elverişsiz: bu pencerede test kullanılamaz
            self.disp_available = True
            old = [self.buf[i0 + i] for i in range(self.n_avg)]
            d = np.mean([b[0] for b in new], axis=0) - np.mean([b[0] for b in old], axis=0)
            C_lio = np.mean([b[1] for b in new], axis=0) - np.mean([b[1] for b in old], axis=0)
            S = S_g + C_lio
            if self.horiz:
                d, S = d[:2], S[:2, :2]
            r = max(r, float(d @ np.linalg.solve(S, d)) / self.thr_d)
        return r

    def step(self, t, nis_p, nis_v, gnss_pos=None, D=None, C=None, counts=(0, 0), **_):
        """Bir GNSS çağında testleri günceller; GNSS kovaryans şişirme çarpanı λ döndürür.

        nis_p/nis_v: normalize yenilik kareleri (geçersiz ölçümde None).
        gnss_pos, D, C: GNSS konumu, LIO birikimli yer değiştirmesi ve kovaryansı (dünya).
        Dönüş: (lambda, info) — lambda=inf ⇒ GNSS dışlanır.
        """
        d = self.d
        dropout = False
        if nis_p is None:
            if self.last_valid_t is not None and t - self.last_valid_t > d.dropout_timeout:
                dropout = True
            return np.inf, self._info(*self.stats(), dropout, alarm=dropout)
        self.last_valid_t = t
        self.win_p.append(nis_p)
        dof = self.k
        s = nis_p
        if self.use_vel and nis_v is not None:
            self.win_v.append(nis_v)
            s += nis_v
            dof += self.k
        self.g = float(np.clip(self.g + s / dof - (1.0 + d.cusum_k), 0.0, self.g_cap))
        if self.use_disp and D is not None:
            self.rD = self._disp_test(gnss_pos, D, C, counts)
        rA, rB, rC, rD = self.stats()
        rho = max(rA, rB, rC, rD)
        votes = int(rA >= 1) + int(rB >= 1) + int(rC >= 1) + int(rD >= 1)
        # teyit: karar ardışık `confirm_epochs` çağ boyunca sürmeli (tek çağlık sıçramalara karşı)
        raw = rho >= 1.0 if self.policy != "voting" else votes >= 2
        self.above = self.above + 1 if raw else 0
        declared = self.above >= d.confirm_epochs
        # bekçi, tolerans süresi denetiminden önce: yeniden kabul edildiği çağda aynı karar GNSS'i
        # hemen yeniden dışlamasın
        self._watchdog(t)
        if t < self.grace_until:
            declared = False
            rho_eff = 0.0
        else:
            rho_eff = rho

        if self.policy in ("monitor", "none"):
            lam = 1.0
            alarm = declared
        elif self.policy in ("hard", "voting"):
            alarm = declared
            if alarm:
                self.excluded = True
                self.consistent_since = None
            elif self.excluded:
                # yeniden kabul: tüm testler belirli süre boyunca tutarlı olmalı
                if rho_eff < 1.0:
                    if self.consistent_since is None:
                        self.consistent_since = t
                    if t - self.consistent_since >= d.readmit_time:
                        self.excluded = False
                        self.g = 0.0
                        self.win_p.clear()
                        self.win_v.clear()
                else:
                    self.consistent_since = None
            lam = np.inf if self.excluded else 1.0
            alarm = alarm or self.excluded
        elif self.policy == "soft":
            tc = self.t
            # testlerin nominal düzeyinin üzerindeki normalize aşım (0: sağlıklı, 1: alarm eşiği)
            oc, ou, od = tc.soft_onset_chi2, tc.soft_onset_cusum, tc.soft_onset_disp
            excess = max((rA - oc) / (1 - oc), (rB - ou) / (1 - ou), (rC - oc) / (1 - oc),
                         (rD - od) / (1 - od), 0.0) if rho_eff > 0 else 0.0
            if declared:
                target = np.log(tc.soft_max_inflation)  # aldatma ilan edildi: fiilen dışlama
            else:
                target = tc.soft_gain * min(excess, 1.0)  # şüphe bölgesi: sürekli şişirme
            # şüphe arttıkça anında, azaldıkça yumuşakça (üstel) geri dönüş
            self.log_lambda = max(target, self.log_lambda + np.log(tc.soft_decay))
            lam = float(np.exp(self.log_lambda))
            alarm = declared
            self.excluded = lam > 1e3
        else:
            raise ValueError(self.policy)
        return lam, self._info(rA, rB, rC, rD, dropout, alarm)

    @staticmethod
    def _info(rA, rB, rC, rD, dropout, alarm):
        return {"rA": rA, "rB": rB, "rC": rC, "rD": rD, "rho": max(rA, rB, rC, rD),
                "dropout": dropout, "alarm": bool(alarm or dropout)}
