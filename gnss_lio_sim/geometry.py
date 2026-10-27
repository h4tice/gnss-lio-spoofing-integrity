"""SO(3) yardımcıları: skew, Exp/Log, kuaterniyon dönüşümleri (vektörleştirilmiş)."""
import numpy as np


def skew(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] = v[..., 1]
    out[..., 1, 0] = v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] = v[..., 0]
    return out


def so3_exp(phi):
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return (np.eye(3) + np.sin(theta) / theta * K
            + (1.0 - np.cos(theta)) / theta ** 2 * K @ K)


def so3_log(R):
    cos_t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * w
    if np.pi - theta < 1e-6:
        # 180° yakınında sayısal olarak kararlı çözüm
        B = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        axis[1] = np.copysign(axis[1], B[0, 1] if axis[0] > 0 else 1.0)
        axis[2] = np.copysign(axis[2], B[0, 2] if axis[0] > 0 else B[1, 2])
        return theta * axis / max(np.linalg.norm(axis), 1e-12)
    return theta / (2.0 * np.sin(theta)) * w


def right_jacobian(phi):
    theta = np.linalg.norm(phi)
    K = skew(phi)
    if theta < 1e-6:
        return np.eye(3) - 0.5 * K
    return (np.eye(3) - (1 - np.cos(theta)) / theta ** 2 * K
            + (theta - np.sin(theta)) / theta ** 3 * K @ K)


def rot_to_quat(R):
    """Dönme matrisi(leri) -> kuaterniyon [w, x, y, z]."""
    R = np.asarray(R)
    single = R.ndim == 2
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4))
    tr = np.trace(R, axis1=1, axis2=2)
    for i in range(R.shape[0]):
        m = R[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        if q[i, 0] < 0:
            q[i] = -q[i]
    return q[0] if single else q


def quat_to_rot(q):
    q = np.asarray(q, dtype=float)
    single = q.ndim == 1
    q = q.reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R[0] if single else R


def yaw_of(R):
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn
