"""Optimal rigid-body superposition and root-mean-square deviation.

Uses the quaternion form of the Kabsch problem (Horn, 1987, *J. Opt. Soc. Am.
A* 4, 629; Coutsias et al., 2004, *J. Comput. Chem.* 25, 1849): the optimal
rotation is the top eigenvector of a 4x4 matrix built from the centred
cross-covariance, which never yields an improper rotation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from . import geometry as geo
from .errors import ComparisonError
from .geometry import Mat3, Vec3


@dataclass(frozen=True)
class Superposition:
    """Result of :func:`kabsch`.

    The transform maps *mobile* coordinates onto the *target* frame:
    ``x_target ~= rotation @ x_mobile + translation``.
    """

    rotation: Mat3
    translation: Vec3
    rmsd: float
    n_points: int

    def apply(self, point: Vec3) -> Vec3:
        """Transform a single mobile-frame point into the target frame."""
        return geo.add(geo.mat_vec(self.rotation, point), self.translation)

    def apply_all(self, points: Sequence[Vec3]) -> list[Vec3]:
        return [self.apply(p) for p in points]


def _check(mobile: Sequence[Vec3], target: Sequence[Vec3], weights: Sequence[float] | None) -> None:
    if len(mobile) != len(target):
        raise ComparisonError(f"point sets differ in size ({len(mobile)} vs {len(target)})")
    if len(mobile) == 0:
        raise ComparisonError("cannot superpose empty point sets")
    if weights is not None:
        if len(weights) != len(mobile):
            raise ComparisonError("weights must match the number of points")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ComparisonError("weights must be non-negative with a positive sum")


def rmsd(a: Sequence[Vec3], b: Sequence[Vec3], weights: Sequence[float] | None = None) -> float:
    """RMSD between two point sets *without* superposition."""
    _check(a, b, weights)
    if weights is None:
        return math.sqrt(sum(geo.distance_sq(p, q) for p, q in zip(a, b)) / len(a))
    total = sum(weights)
    return math.sqrt(sum(w * geo.distance_sq(p, q) for w, p, q in zip(weights, a, b)) / total)


def quaternion_to_matrix(q: Sequence[float]) -> Mat3:
    """Convert a unit quaternion ``(w, x, y, z)`` to a rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z),
    )


def kabsch(mobile: Sequence[Vec3], target: Sequence[Vec3], weights: Sequence[float] | None = None) -> Superposition:
    """Rigid transform minimising the (optionally weighted) RMSD of *mobile* onto *target*."""
    _check(mobile, target, weights)
    w = list(weights) if weights is not None else [1.0] * len(mobile)
    cm = geo.centroid(mobile, w)
    ct = geo.centroid(target, w)
    sxx = sxy = sxz = syx = syy = syz = szx = szy = szz = 0.0
    for wi, p, q in zip(w, mobile, target):
        px, py, pz = p[0] - cm[0], p[1] - cm[1], p[2] - cm[2]
        qx, qy, qz = q[0] - ct[0], q[1] - ct[1], q[2] - ct[2]
        sxx += wi * px * qx
        sxy += wi * px * qy
        sxz += wi * px * qz
        syx += wi * py * qx
        syy += wi * py * qy
        syz += wi * py * qz
        szx += wi * pz * qx
        szy += wi * pz * qy
        szz += wi * pz * qz
    key = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    _, vectors = geo.jacobi_eigh(key)
    quat = [vectors[i][0] for i in range(4)]
    rotation = quaternion_to_matrix(quat)
    translation = geo.sub(ct, geo.mat_vec(rotation, cm))
    moved = [geo.add(geo.mat_vec(rotation, p), translation) for p in mobile]
    return Superposition(rotation, translation, rmsd(moved, target, weights), len(mobile))


def superposed_rmsd(mobile: Sequence[Vec3], target: Sequence[Vec3], weights: Sequence[float] | None = None) -> float:
    """Minimum RMSD after optimal superposition (shortcut for ``kabsch(...).rmsd``)."""
    return kabsch(mobile, target, weights).rmsd
