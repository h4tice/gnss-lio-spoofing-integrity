"""LiDAR-ataletsel odometri (LIO): FAST-LIO2 esinli sıkı bağlı IEKF.

* IMU ile durum yayılımı (ins.Ins)
* Her taramada, harita üzerindeki k-en-yakın komşulardan düzlem uydurma ve
  nokta-düzlem artıklarıyla yinelemeli hata-durumlu Kalman güncellemesi
* Vokselleştirilmiş artımlı harita + periyodik yerel kd-ağacı yenileme
* Çıktı: poz, göreli poz kovaryansı (füzyon için) ve ortam geometrisi metrikleri
  (düzlem/kenar öznitelik yoğunluğu, kısıt bilgi matrisi özdeğerleri) — AS3 için.
"""
import time

import numpy as np
from scipy.spatial import cKDTree

from .geometry import so3_exp, so3_log
from .ins import Ins, N_CORE

POSE_IDX = np.r_[0:3, 6:9]


def voxel_downsample(pts, voxel):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


class VoxelMap:
    def __init__(self, voxel, radius, rebuild_every):
        self.voxel = voxel
        self.radius = radius
        self.rebuild_every = rebuild_every
        self.keys = set()
        self.chunks = []
        self.all_pts = np.zeros((0, 3))
        self.tree = None
        self.tree_pts = None
        self.counter = 0

    def __len__(self):
        return len(self.keys)

    def add(self, pts_w):
        k = np.floor(pts_w / self.voxel).astype(np.int64)
        h = (k[:, 0] + (1 << 20)) | ((k[:, 1] + (1 << 20)) << 21) | ((k[:, 2] + (1 << 20)) << 42)
        h, first = np.unique(h, return_index=True)
        lst = h.tolist()
        keys = self.keys
        mask = np.fromiter((x not in keys for x in lst), dtype=bool, count=len(lst))
        if mask.any():
            keys.update(h[mask].tolist())
            self.chunks.append(pts_w[first[mask]])

    def maybe_rebuild(self, center, force=False):
        self.counter += 1
        if not force and self.tree is not None and self.counter % self.rebuild_every:
            return
        if self.chunks:
            self.all_pts = np.vstack([self.all_pts] + self.chunks)
            self.chunks = []
        d = np.linalg.norm(self.all_pts[:, :2] - center[:2], axis=1)
        self.tree_pts = self.all_pts[d < self.radius]
        self.tree = cKDTree(self.tree_pts) if len(self.tree_pts) > 10 else None

    def query(self, pts, k, max_dist):
        d, i = self.tree.query(pts, k=k, distance_upper_bound=max_dist)
        valid = np.isfinite(d).all(axis=1)
        i = np.where(valid[:, None], i, 0)
        return self.tree_pts[i], valid


def scan_geometry(pts_b, k=8):
    """Tarama içi yerel geometri: kenar (doğrusal) ve düzlemsel nokta sayıları."""
    if len(pts_b) < k + 1:
        return 0, 0
    tree = cKDTree(pts_b)
    _, idx = tree.query(pts_b, k=k)
    nb = pts_b[idx]
    A = nb - nb.mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", A, A) / k
    ev = np.linalg.eigvalsh(C)[:, ::-1]  # λ1 ≥ λ2 ≥ λ3
    l1 = np.maximum(ev[:, 0], 1e-9)
    lin = (ev[:, 0] - ev[:, 1]) / l1
    pla = (ev[:, 1] - ev[:, 2]) / l1
    return int((lin > 0.8).sum()), int((pla > 0.6).sum())


def degeneracy_flags(pos_info_min, cfg, rot_info_min=None):
    """Nedensel kayan medyan (son ~2 s) en küçük konum bilgisi özdeğeri eşik altındaysa
    tarama yatayda dejenere sayılır (tekil taramalardaki kuyruk değerlerine karşı dayanıklı)."""
    import pandas as pd
    med = pd.Series(pos_info_min).rolling(cfg.deg_window, min_periods=1).median().values
    deg = med < cfg.deg_horiz_thresh
    if rot_info_min is not None:
        # yaw gözlenemiyorsa LIO yatay yer değiştirmenin yönünü de doğrulayamaz
        rmed = pd.Series(rot_info_min).rolling(cfg.deg_window, min_periods=1).median().values
        deg |= rmed < cfg.deg_yaw_thresh
    return deg


def relative_cov(info6, R_prev, cfg, degenerate=False):
    """LIO tarama-kayıt bilgi matrisinden göreli poz ölçüm kovaryansı (önceki gövde çerçevesinde).

    Nokta-düzlem bilgisi muhafazakâr ölçeklenir (harita/ilintili hatalar için);
    dejenere yönlerde önsel belirsizlik baskın olur, böylece füzyon filtresi o
    yönlerde LIO'ya güvenmez (ör. yapısal olarak fakir düz alan).
    """
    # dejenerelik faktörü (Zhang ve ark., 2016): eşik altı öz-yönlerdeki bilgi atılır.
    # Ek olarak, yapısal kısıt zayıfsa (degeneracy_flags) yatay konum ve yaw yönleri
    # dejenere sayılır (hafif eğimli zeminden gelen zayıf ve yanlı bilgi).
    no_struct = bool(degenerate)
    Pj = np.zeros((6, 6))
    for sl, thr, is_pos in ((slice(0, 3), cfg.deg_pos_thresh, True),
                            (slice(3, 6), cfg.deg_rot_thresh, False)):
        ev, V = np.linalg.eigh(info6[sl, sl])
        keep = ev >= thr
        if no_struct:
            vertical = np.abs(V[2]) > 0.7
            keep &= vertical if is_pos else ~vertical  # düşey öteleme ve roll/pitch korunur
        Vk = V[:, keep]
        Pj[sl, sl] = Vk @ Vk.T
    info = Pj @ info6 @ Pj
    prior = np.r_[np.full(3, 1 / cfg.rel_pos_prior ** 2),
                  np.full(3, 1 / np.deg2rad(cfg.rel_rot_prior_deg) ** 2)]
    cov = np.linalg.inv(cfg.rel_info_scale * info + np.diag(prior))
    cov += np.diag(np.r_[np.full(3, cfg.rel_pos_floor ** 2),
                         np.full(3, np.deg2rad(cfg.rel_rot_floor_deg) ** 2)])
    cov[:3, :3] = R_prev.T @ cov[:3, :3] @ R_prev
    cov[:3, 3:] = R_prev.T @ cov[:3, 3:]
    cov[3:, :3] = cov[:3, 3:].T
    return cov


class LIO:
    def __init__(self, cfg, imu_cfg, p0, R0, v0=None):
        self.cfg = cfg
        P0 = Ins.default_P0(sig_p=0.01, sig_v=0.05, sig_th_deg=0.1, sig_ba=0.05, sig_bg=0.005)
        self.ins = Ins(p0, np.zeros(3) if v0 is None else v0, R0, P0, imu_cfg)
        self.map = VoxelMap(cfg.map_voxel, cfg.map_radius, cfg.kdtree_rebuild_every)
        self.rng = np.random.default_rng(0)
        self.sigma2 = cfg.meas_sigma ** 2
        self.match_dist = cfg.get("match_dist", 1.5)
        self.last_gyro = np.zeros(3)

    def propagate(self, acc, gyro, dt):
        self.ins.propagate(acc, gyro, dt)
        self.last_gyro = np.asarray(gyro, float)

    def deskew(self, pts, dt_to_end):
        """Hareket bozulması düzeltmesi: her noktayı tarama sonu gövde çerçevesine taşır
        (tarama süresince sabit açısal hız ve hız varsayımı)."""
        ins = self.ins
        w = self.last_gyro - ins.bg
        d = dt_to_end[:, None]
        phi = -w[None, :] * d
        th = np.linalg.norm(phi, axis=1, keepdims=True)
        k = phi / np.maximum(th, 1e-12)
        c, s = np.cos(th), np.sin(th)
        rot = pts * c + np.cross(k, pts) * s + k * np.sum(k * pts, axis=1, keepdims=True) * (1 - c)
        return rot - d * (ins.R.T @ ins.v)[None, :]

    def _residuals(self, pts, R, p):
        cfg = self.cfg
        pw = pts @ R.T + p
        nb, valid = self.map.query(pw, cfg.knn, self.match_dist)
        c = nb.mean(axis=1)
        A = nb - c[:, None, :]
        C = np.einsum("nki,nkj->nij", A, A)
        ev, vec = np.linalg.eigh(C)
        n = vec[:, :, 0]
        planar = np.all(np.abs(np.einsum("nki,ni->nk", A, n)) < cfg.plane_thresh, axis=1)
        # seyrek halkalı LiDAR'da (VLP-16) komşular tek bir halka çizgisinde olabilir: doğrusal küme düzlem
        # tanımlamaz, normali rastgeledir; ikinci özdeğer en büyüğüne göre küçükse eşleşme atılır
        planar &= ev[:, 1] >= cfg.get("plane_min_ratio", 0.0) * ev[:, 2]
        r = np.einsum("ni,ni->n", n, pw - c)
        rn = np.linalg.norm(pts, axis=1)
        s = 1.0 - 0.9 * np.abs(r) / np.sqrt(np.maximum(rn, 1e-3))  # FAST-LIO uzaklık ağırlığı
        ok = valid & planar & (s > 0.9) & (np.abs(r) < 0.5)
        return n[ok], r[ok], pts[ok]

    def update(self, t, scan_b, scan_dt=None):
        """Bir LiDAR taramasıyla IEKF güncellemesi. Çıktı kaydı (dict) döndürür.

        scan_dt verilirse (noktanın tarama sonuna kalan süresi [s]) önce hareket bozulması düzeltilir.
        """
        cfg = self.cfg
        tic = time.perf_counter()
        ins = self.ins
        if scan_dt is not None:
            scan_b = self.deskew(scan_b.astype(float), scan_dt.astype(float))
        pts = voxel_downsample(scan_b.astype(float), cfg.scan_voxel)
        if len(pts) > cfg.max_points:
            pts = pts[self.rng.choice(len(pts), cfg.max_points, replace=False)]

        info6 = np.zeros((6, 6))
        normals = np.zeros((0, 3))
        n_corr = 0
        if self.map.tree is not None:
            x0 = (ins.p.copy(), ins.v.copy(), ins.R.copy(), ins.ba.copy(), ins.bg.copy())
            P0 = ins.P[:N_CORE, :N_CORE].copy()
            P0inv = np.linalg.inv(P0)
            Sinv_last = None
            for _ in range(cfg.max_iter):
                n, r, pb = self._residuals(pts, ins.R, ins.p)
                n_corr = len(r)
                if n_corr < 30:
                    break
                Hth = -np.cross(n @ ins.R, pb)
                H6 = np.hstack([n, Hth])
                info6 = H6.T @ H6 / self.sigma2
                Info = np.zeros((N_CORE, N_CORE))
                Info[np.ix_(POSE_IDX, POSE_IDX)] = info6
                Htr = np.zeros(N_CORE)
                Htr[POSE_IDX] = H6.T @ r / self.sigma2
                dxp = np.r_[ins.p - x0[0], ins.v - x0[1], so3_log(x0[2].T @ ins.R),
                            ins.ba - x0[3], ins.bg - x0[4]]
                Sinv = np.linalg.inv(Info + P0inv)
                dx = -Sinv @ Htr - (np.eye(N_CORE) - Sinv @ Info) @ dxp
                ins.inject(dx)
                Sinv_last = Sinv
                normals = n
                if np.linalg.norm(dx[0:3]) < 1e-3 and np.linalg.norm(dx[6:9]) < 1e-4:
                    break
            if Sinv_last is not None:
                ins.P = Sinv_last
        # haritaya ekle
        pw_full = voxel_downsample(scan_b.astype(float), cfg.map_voxel) @ ins.R.T + ins.p
        self.map.add(pw_full)
        self.map.maybe_rebuild(ins.p, force=self.map.tree is None)
        dt_proc = time.perf_counter() - tic

        # ---- geometri metrikleri (AS3)
        n_edge, n_plane_scan = scan_geometry(pts)
        if len(normals) > 0:
            Nn = normals.T @ normals / len(normals)
            nev = np.linalg.eigvalsh(Nn)
            n_vert = int((np.abs(normals[:, 2]) < 0.5).sum())
        else:
            nev = np.zeros(3)
            n_vert = 0
        pos_info_ev = np.linalg.eigvalsh(info6[:3, :3])
        rot_info_ev = np.linalg.eigvalsh(info6[3:, 3:])

        return {
            "t": t, "p": ins.p.copy(), "R": ins.R.copy(), "v": ins.v.copy(),
            "info6": info6, "n_corr": n_corr, "n_edge": n_edge, "n_plane_scan": n_plane_scan,
            "n_vert": n_vert, "normal_ev": nev, "pos_info_ev": pos_info_ev,
            "rot_info_ev": rot_info_ev, "n_scan": len(scan_b), "proc_time": dt_proc,
            "map_size": len(self.map),
        }
