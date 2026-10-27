"""3B deney ortamları (üçgen ağ) ve LiDAR için ışın izleme sahnesi.

Önerideki üç ortam tipi:
  * open_field : açık arazi  - engebeli zemin, seyrek ağaç/direk/çit
  * campus     : yarı yapılı kampüs - binalar, duvarlar, direkler, araçlar, ağaçlar
  * flat_poor  : yapısal olarak fakir düz alan - neredeyse düz zemin, menzil içinde
                 hemen hiç dikey yapı yok (LIO için dejenere durum)

Engeller uçuş yolundan (xy düzleminde) güvenli bir mesafede tutulur.
"""
import numpy as np
import open3d as o3d


# ----------------------------------------------------------------- yardımcılar
def _terrain(size, res, amp, rng, n_waves=6, wl=(25.0, 90.0)):
    """Rastgele sinüs toplamından yumuşak arazi yükseklik fonksiyonu."""
    k = rng.uniform(2 * np.pi / wl[1], 2 * np.pi / wl[0], n_waves)
    ang = rng.uniform(0, 2 * np.pi, n_waves)
    ph = rng.uniform(0, 2 * np.pi, n_waves)
    a = rng.uniform(0.5, 1.0, n_waves)
    a = amp * a / a.sum()
    kx, ky = k * np.cos(ang), k * np.sin(ang)

    def height(x, y):
        x = np.asarray(x)[..., None]
        y = np.asarray(y)[..., None]
        return np.sum(a * np.sin(kx * x + ky * y + ph), axis=-1)

    half = size / 2
    xs = np.arange(-half, half + 1e-6, res)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = height(X, Y)
    n = len(xs)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a0, a1 = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    b0, b1 = idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a0, a1, b1], 1), np.stack([a0, b1, b0], 1)])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(tris))
    return mesh, height


def _box(cx, cy, z0, sx, sy, sz, yaw=0.0):
    m = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
    m.translate((-sx / 2, -sy / 2, 0.0))
    m.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw)), center=(0, 0, 0))
    m.translate((cx, cy, z0))
    return m


def _cylinder(cx, cy, z0, r, h, res=12):
    m = o3d.geometry.TriangleMesh.create_cylinder(r, h, resolution=res, split=2)
    m.translate((cx, cy, z0 + h / 2))
    return m


def _sphere(cx, cy, cz, r, res=8):
    m = o3d.geometry.TriangleMesh.create_sphere(r, resolution=res)
    m.translate((cx, cy, cz))
    return m


def _cone(cx, cy, z0, r, h, res=12):
    m = o3d.geometry.TriangleMesh.create_cone(r, h, resolution=res, split=1)
    m.translate((cx, cy, z0))
    return m


def _tree(rng, x, y, z0, scale=1.0):
    parts = []
    trunk_h = rng.uniform(2.0, 3.5) * scale
    parts.append(_cylinder(x, y, z0, rng.uniform(0.15, 0.3) * scale, trunk_h, res=8))
    if rng.random() < 0.5:
        parts.append(_sphere(x, y, z0 + trunk_h + 1.2 * scale, rng.uniform(1.3, 2.5) * scale))
    else:
        parts.append(_cone(x, y, z0 + trunk_h * 0.6, rng.uniform(1.2, 2.0) * scale,
                           rng.uniform(3.0, 5.0) * scale))
    return parts


def _dist_to_path(xy, path_xy):
    """Noktaların yol çoklu-doğrusuna en kısa mesafesi (kaba, yoğun örneklenmiş yol)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(path_xy)
    d, _ = tree.query(np.atleast_2d(xy))
    return d


class Environment:
    def __init__(self, env_type, mesh, height_fn, meta):
        self.env_type = env_type
        self.mesh = mesh
        self.height = height_fn
        self.meta = meta
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def cast(self, origins, dirs):
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        return ans["t_hit"].numpy()


# ----------------------------------------------------------------- ortam kurucular
def build_environment(env_type, size, path_xy, rng):
    """`path_xy`: (N,2) yoğun örneklenmiş uçuş yolu; engeller bu yoldan uzak tutulur."""
    meshes = []
    objects = []
    clearance = 7.0

    def free(x, y, radius):
        return _dist_to_path(np.array([[x, y]]), path_xy)[0] > clearance + radius

    def scatter(n, radius, margin=8.0, tries=40):
        pts = []
        half = size / 2 - margin
        for _ in range(n):
            for _ in range(tries):
                x, y = rng.uniform(-half, half, 2)
                if free(x, y, radius) and all(np.hypot(x - px, y - py) > radius + pr + 1.0
                                              for px, py, pr in pts):
                    pts.append((x, y, radius))
                    break
        return pts

    if env_type == "open_field":
        ground, h = _terrain(size, 2.0, amp=1.2, rng=rng)
        meshes.append(ground)
        for x, y, r in scatter(45, 2.5):
            meshes += _tree(rng, x, y, h(x, y) - 0.2, scale=rng.uniform(0.8, 1.3))
            objects.append(("tree", x, y))
        # çit hattı (direkler)
        y0 = rng.uniform(-size / 3, size / 3)
        for x in np.arange(-size / 2 + 10, size / 2 - 10, 6.0):
            if free(x, y0, 0.3) and rng.random() < 0.8:
                meshes.append(_cylinder(x, y0, h(x, y0) - 0.2, 0.08, 1.6, res=6))
        # birkaç kaya / çalı
        for x, y, r in scatter(25, 1.0):
            meshes.append(_sphere(x, y, h(x, y), rng.uniform(0.5, 1.0), res=6))
        # küçük bir depo
        for x, y, r in scatter(2, 5.0):
            meshes.append(_box(x, y, h(x, y) - 0.3, rng.uniform(5, 8), rng.uniform(4, 6),
                               rng.uniform(3, 4), yaw=rng.uniform(0, np.pi)))
            objects.append(("shed", x, y))

    elif env_type == "campus":
        ground, h = _terrain(size, 2.0, amp=0.25, rng=rng)
        meshes.append(ground)
        for x, y, r in scatter(30, 9.0, tries=80):
            sx, sy = rng.uniform(8, 22), rng.uniform(8, 22)
            sx, sy = min(sx, 2 * r), min(sy, 2 * r)
            meshes.append(_box(x, y, h(x, y) - 0.5, sx, sy, rng.uniform(6, 20),
                               yaw=rng.choice([0.0, np.pi / 2, rng.uniform(0, np.pi)])))
            objects.append(("building", x, y))
        for x, y, r in scatter(20, 4.0):  # alçak duvarlar
            meshes.append(_box(x, y, h(x, y) - 0.2, rng.uniform(4, 12), 0.3,
                               rng.uniform(1.0, 2.2), yaw=rng.uniform(0, np.pi)))
        for x, y, r in scatter(60, 0.5):  # aydınlatma direkleri
            meshes.append(_cylinder(x, y, h(x, y) - 0.2, 0.12, rng.uniform(4, 7), res=6))
        for x, y, r in scatter(25, 2.5):  # park halindeki araçlar
            meshes.append(_box(x, y, h(x, y), 4.3, 1.8, 1.5, yaw=rng.uniform(0, np.pi)))
        for x, y, r in scatter(35, 2.5):
            meshes += _tree(rng, x, y, h(x, y) - 0.2)

    elif env_type == "flat_poor":
        ground, h = _terrain(size, 2.0, amp=0.15, rng=rng, wl=(40.0, 120.0))
        meshes.append(ground)
        # menzil dışında kalan çok seyrek uzak yapılar
        half = size / 2 - 3
        for _ in range(3):
            x, y = rng.choice([-half, half]), rng.uniform(-half, half)
            if rng.random() < 0.5:
                x, y = y, x
            meshes.append(_box(x, y, 0.0, 4, 4, 3))
    else:
        raise ValueError(f"bilinmeyen ortam tipi: {env_type}")

    mesh = meshes[0]
    for m in meshes[1:]:
        mesh += m
    mesh.compute_vertex_normals()
    meta = {"type": env_type, "objects": len(objects), "triangles": len(mesh.triangles)}
    return Environment(env_type, mesh, h, meta)
