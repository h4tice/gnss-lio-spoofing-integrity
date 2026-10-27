"""Open3D ile 3B uçuş oynatma: ortam ağı, anlık LiDAR taraması, gerçek/kestirilen
yörünge, aldatılmış GNSS ölçümleri ve drone gövde çerçevesi.

Kısayollar (Open3D penceresi):  SPACE duraklat/devam · +/- hız · R baştan · Q/ESC çıkış
"""
import numpy as np
import open3d as o3d

from .geometry import quat_to_rot

COL_TRUTH = (0.32, 0.32, 0.31)
COL_EST = (0.106, 0.686, 0.478)
COL_LIO = (0.0, 0.514, 0.0)
COL_GNSS = (0.929, 0.631, 0.0)
COL_SCAN = (0.165, 0.471, 0.839)


def _lineset(pts, color):
    ls = o3d.geometry.LineSet()
    if len(pts) >= 2:
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(np.c_[np.arange(len(pts) - 1), np.arange(1, len(pts))])
        ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts) - 1, 1)))
    return ls


def replay(env, flight, scans, fused, gnss_att, title="GNSS-LIO benzetimi", speed=2.0):
    t_scan = flight["lio_t"]
    truth_idx = np.searchsorted(flight["truth_t"], t_scan - 1e-6)
    p_true = flight["truth_p"][truth_idx]
    R_true = quat_to_rot(flight["truth_q"][truth_idx])
    p_est = fused["p"]
    p_lio = flight["lio_p"]
    g_t, g_pos, g_valid = gnss_att["t"], gnss_att["pos"], gnss_att["valid"]
    lam = np.interp(t_scan, fused["g_t"], fused["g_lam"]) if len(fused["g_t"]) else np.ones(len(t_scan))
    t_on, t_off = gnss_att["window"]

    mesh = o3d.geometry.TriangleMesh(env.mesh)
    mesh.paint_uniform_color((0.80, 0.79, 0.76))
    mesh.compute_vertex_normals()
    scan_pc = o3d.geometry.PointCloud()
    gnss_pc = o3d.geometry.PointCloud()
    tr_ls, est_ls, lio_ls = _lineset(p_true[:2], COL_TRUTH), _lineset(p_est[:2], COL_EST), _lineset(p_lio[:2], COL_LIO)
    body = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    body_verts0 = np.asarray(body.vertices).copy()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(title, width=1400, height=900)
    for g in (mesh, scan_pc, gnss_pc, tr_ls, est_ls, lio_ls, body):
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.line_width = 3.0
    opt.background_color = np.array([0.98, 0.98, 0.97])

    state = {"k": 0, "paused": False, "speed": speed, "follow": True}

    def toggle(_):
        state["paused"] = not state["paused"]
        return False

    def faster(_):
        state["speed"] = min(state["speed"] * 1.5, 50)
        return False

    def slower(_):
        state["speed"] = max(state["speed"] / 1.5, 0.25)
        return False

    def restart(_):
        state["k"] = 0
        return False

    vis.register_key_callback(32, toggle)
    vis.register_key_callback(ord("="), faster)
    vis.register_key_callback(ord("+"), faster)
    vis.register_key_callback(ord("-"), slower)
    vis.register_key_callback(ord("R"), restart)

    last = {"k": -1}

    def step(v):
        if state["paused"]:
            return False
        state["k"] = min(state["k"] + max(1, int(round(state["speed"]))), len(t_scan) - 1)
        k = state["k"]
        if k == last["k"]:
            return False
        last["k"] = k
        R, p = R_true[k], p_true[k]
        scan_pc.points = o3d.utility.Vector3dVector(scans[k] @ R.T + p)
        scan_pc.paint_uniform_color(COL_SCAN)
        m = (g_t <= t_scan[k]) & g_valid
        gnss_pc.points = o3d.utility.Vector3dVector(g_pos[m])
        gnss_pc.paint_uniform_color(COL_GNSS)
        for ls, pts, col in ((tr_ls, p_true, COL_TRUTH), (est_ls, p_est, COL_EST), (lio_ls, p_lio, COL_LIO)):
            new = _lineset(pts[:k + 1], col)
            ls.points, ls.lines, ls.colors = new.points, new.lines, new.colors
        body.vertices = o3d.utility.Vector3dVector(body_verts0 @ R.T + p)
        for g in (scan_pc, gnss_pc, tr_ls, est_ls, lio_ls, body):
            v.update_geometry(g)
        if state["follow"]:
            ctr = v.get_view_control()
            ctr.set_lookat(p)
        if k % 10 == 0:
            phase = "SALDIRI" if t_on <= t_scan[k] < t_off else "normal"
            err = np.linalg.norm(p_est[min(k, len(p_est) - 1)] - p)
            print(f"\r t={t_scan[k]:6.1f}s  [{phase:7s}]  hata={err:5.2f} m  GNSS şişirme λ={lam[k]:9.1f}   ",
                  end="", flush=True)
        return True

    vis.register_animation_callback(step)
    ctr = vis.get_view_control()
    ctr.set_lookat(p_true[0])
    ctr.set_front((-0.5, -0.6, 0.62))
    ctr.set_up((0, 0, 1))
    ctr.set_zoom(0.12)
    print("Renkler: gri=gerçek yörünge, yeşil-turkuaz=füzyon kestirimi, koyu yeşil=yalnız LIO, "
          "sarı=GNSS ölçümleri, mavi=LiDAR taraması.  SPACE: duraklat, +/-: hız, R: baştan")
    vis.run()
    vis.destroy_window()
    print()
