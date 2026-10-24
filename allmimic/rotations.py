"""Differentiable rotation helpers. Quaternions are (w, x, y, z), matching MuJoCo."""

from __future__ import annotations

import torch

# Heading convention (set once by Task from TaskCfg.heading, before anything is compiled):
#   'xaxis'  yaw of the body x axis projected on the ground (upstream); singular when the pelvis x axis is vertical,
#            i.e. lying, crawling, handstands -- jumps up to 170 deg between LAFAN1 frames (CHANGES #14)
#   'twist'  swing-twist yaw atan2(R10 - R01, R00 + R11) = 2 atan2(qz, qw); singular only fully upside down
HEADING = {'mode': 'xaxis'}

# 3x3 products as broadcast multiply-add: torch.compile fuses these, whereas `@`
# on batches of 3x3 matrices becomes many small bmm calls.


def mv(R, x):
    """R x"""
    return (R * x.unsqueeze(-2)).sum(-1)


def mtv(R, x):
    """R^T x"""
    return (R * x.unsqueeze(-1)).sum(-2)


def mm(A, B):
    """A B"""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def mtm(A, B):
    """A^T B"""
    return (A.unsqueeze(-1) * B.unsqueeze(-2)).sum(-3)


def mmt(A, B):
    """A B^T"""
    return (A.unsqueeze(-2) * B.unsqueeze(-3)).sum(-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack(
        (
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ),
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to quaternion (w >= 0). Not used inside gradient paths."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt((1 + m00 + m11 + m22).clamp_min(1e-12)) / 2
    qx = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) / 2
    qy = torch.sqrt((1 - m00 + m11 - m22).clamp_min(1e-12)) / 2
    qz = torch.sqrt((1 - m00 - m11 + m22).clamp_min(1e-12)) / 2
    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    return quat_normalize(torch.stack((qw, qx, qy, qz), dim=-1))


def axis_angle_to_mat(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula, smooth at zero."""
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(v.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
    return eye + a[..., None] * K + b[..., None] * mm(K, K)


def mat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Log map SO(3) -> R^3, differentiable away from theta = pi."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos)
    w = torch.stack(
        (R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]),
        dim=-1,
    )
    sin = torch.sin(theta)
    small = theta < 1e-3
    scale = torch.where(small, 0.5 + theta * theta / 12, theta / (2 * sin.clamp_min(1e-6)))
    return w * scale[..., None]


def rot6d(R: torch.Tensor) -> torch.Tensor:
    """First two columns of a rotation matrix, flattened (Zhou et al. 2019)."""
    return R[..., :, :2].transpose(-1, -2).reshape(R.shape[:-2] + (6,))


# Tri-point code (a tri-point local representation, CHANGES #18 / #25): three marker points of a
# centred unit triangle carried by the frame, marker m = origin + R @ (e_m - mean(e)). The centred markers sum to zero,
# so the 9 numbers hold the 6 rotational DoF plus the origin (= the marker mean); decoded by Kabsch / SVD.
TRI_REF = torch.eye(3) - 1.0 / 3.0  # (3 markers, 3)


def _tri_ref(x):
    """TRI_REF built on x's device / dtype: TRI_REF.to(x) was a blocking host->device copy on every call (one host
    sync per eager call; a CPU input inside compiled graphs, which also keeps cudagraphs off). Same values."""
    return (torch.eye(3, device=x.device) - 1.0 / 3.0).to(x.dtype)  # float32 first, exactly like TRI_REF


def tripoint(R: torch.Tensor, origin: torch.Tensor | None = None) -> torch.Tensor:
    """(...,3,3) rotation [, (...,3) origin] -> (...,9) markers [m0, m1, m2], m = origin + R @ ref_m"""
    m = (R.unsqueeze(-3) * _tri_ref(R)[:, None, :]).sum(-1)  # (...,3 markers,3)
    if origin is not None:
        m = m + origin.unsqueeze(-2)
    return m.flatten(-2)


def tripoint_decode(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(...,9) markers -> ((...,3,3) rotation by Kabsch / SVD, (...,3) origin = marker mean)"""
    m = x.unflatten(-1, (3, 3))
    origin = m.mean(-2)
    Hm = _tri_ref(x).T @ (m - origin.unsqueeze(-2))  # (...,3,3) = sum_m ref_m centred_m^T
    U, _, Vt = torch.linalg.svd(Hm)
    d = torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2))
    S = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), torch.sign(d)], -1))
    return Vt.transpose(-1, -2) @ S @ U.transpose(-1, -2), origin


def _gs_frame(u, v):
    """(...,3) u, v -> (...,3,3) orthonormal frame [u, v orthogonalized, u x v] as columns"""
    a = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = v - (a * v).sum(-1, keepdim=True) * a
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], dim=-1)


def tripoint_decode_gs(x: torch.Tensor) -> torch.Tensor:
    """(...,9) markers -> (...,3,3) rotation from the marker differences m0 - m1 = R (e0 - e1), m0 - m2 = R (e0 - e2)
    by Gram-Schmidt: exact for the markers of a rotation, smooth and differentiable (the Kabsch decode's SVD has two
    equal singular values for these markers, so its backward divides by zero). The marker mean is ignored."""
    m = x.unflatten(-1, (3, 3))
    e = torch.eye(3, device=x.device, dtype=x.dtype)
    Fr = _gs_frame(e[0] - e[1], e[0] - e[2])
    return _gs_frame(m[..., 0, :] - m[..., 1, :], m[..., 0, :] - m[..., 2, :]) @ Fr.T


def yaw_from_mat(R: torch.Tensor) -> torch.Tensor:
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def yaw_mat(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(yaw.shape + (3, 3))


def heading_mat(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) body rotation -> (...,3,3) yaw-only rotation (HEADING['mode'] picks the convention)"""
    if HEADING['mode'] == 'twist':
        return yaw_mat(torch.atan2(R[..., 1, 0] - R[..., 0, 1], R[..., 0, 0] + R[..., 1, 1]))
    return yaw_mat(yaw_from_mat(R))


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ),
        dim=-1,
    )


def quat_slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    d = (a * b).sum(-1, keepdim=True)
    b = torch.where(d < 0, -b, b)
    d = d.abs().clamp(max=1.0)
    theta = torch.acos(d)
    s = torch.sin(theta)
    t = t[..., None] if t.dim() < a.dim() else t
    wa = torch.where(s > 1e-6, torch.sin((1 - t) * theta) / s.clamp_min(1e-6), 1 - t)
    wb = torch.where(s > 1e-6, torch.sin(t * theta) / s.clamp_min(1e-6), t)
    return quat_normalize(wa * a + wb * b)
