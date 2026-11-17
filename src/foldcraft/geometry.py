"""Small, dependency-free linear algebra for three-dimensional coordinates.

Plain tuples keep the package free of runtime dependencies; hot loops
elsewhere inline the arithmetic where it matters.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ORIGIN: Vec3 = (0.0, 0.0, 0.0)


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def normalize(a: Vec3) -> Vec3:
    """Unit vector parallel to *a*; raises :class:`ValueError` for a zero vector."""
    n = norm(a)
    if n == 0.0:
        raise ValueError("cannot normalise a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def distance(a: Vec3, b: Vec3) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def distance_sq(a: Vec3, b: Vec3) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def angle(a: Vec3, b: Vec3, c: Vec3) -> float:
    """Planar angle ``a-b-c`` in degrees."""
    u, v = sub(a, b), sub(c, b)
    denom = norm(u) * norm(v)
    if denom == 0.0:
        raise ValueError("angle undefined for coincident points")
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(u, v) / denom))))


def dihedral(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float:
    """Signed torsion angle ``a-b-c-d`` in degrees, in the range (-180, 180].

    Uses the numerically stable ``atan2`` formulation (Blondel & Karplus, 1996).
    """
    b0, b1, b2 = sub(a, b), sub(c, b), sub(d, c)
    n1 = norm(b1)
    if n1 == 0.0:
        raise ValueError("dihedral undefined: central bond has zero length")
    b1u = scale(b1, 1.0 / n1)
    v = sub(b0, scale(b1u, dot(b0, b1u)))
    w = sub(b2, scale(b1u, dot(b2, b1u)))
    result = math.degrees(math.atan2(dot(cross(b1u, v), w), dot(v, w)))
    return 180.0 if result == -180.0 else result


def centroid(points: Iterable[Vec3], weights: Sequence[float] | None = None) -> Vec3:
    """(Weighted) arithmetic mean of a collection of points."""
    pts = list(points)
    if not pts:
        raise ValueError("centroid of an empty point set is undefined")
    w = [1.0] * len(pts) if weights is None else [float(x) for x in weights]
    if len(w) != len(pts):
        raise ValueError("weights and points differ in length")
    total = sum(w)
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return tuple(sum(wi * p[k] for wi, p in zip(w, pts)) / total for k in range(3))  # type: ignore[return-value]


def mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return (dot(m[0], v), dot(m[1], v), dot(m[2], v))


def determinant(m: Mat3) -> float:
    return dot(m[0], cross(m[1], m[2]))


def rotation_about_axis(axis: Vec3, degrees: float) -> Mat3:
    """Rotation matrix for a right-handed rotation about *axis* (Rodrigues)."""
    x, y, z = normalize(axis)
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    k = 1.0 - c
    return (
        (c + x * x * k, x * y * k - z * s, x * z * k + y * s),
        (y * x * k + z * s, c + y * y * k, y * z * k - x * s),
        (z * x * k - y * s, z * y * k + x * s, c + z * z * k),
    )


def place_atom(a: Vec3, b: Vec3, c: Vec3, bond: float, theta: float, phi: float) -> Vec3:
    """Place atom ``d`` with bond ``|cd|``, angle ``b-c-d`` and torsion ``a-b-c-d`` (degrees).

    Implements the NeRF construction of Parsons et al. (2005), *J. Comput.
    Chem.* 26, 1063.
    """
    bc = normalize(sub(c, b))
    n = normalize(cross(sub(b, a), bc))
    m = cross(n, bc)
    th = math.radians(theta)
    ph = math.radians(phi)
    d2 = (
        -bond * math.cos(th),
        bond * math.sin(th) * math.cos(ph),
        bond * math.sin(th) * math.sin(ph),
    )
    return (
        c[0] + bc[0] * d2[0] + m[0] * d2[1] + n[0] * d2[2],
        c[1] + bc[1] * d2[0] + m[1] * d2[1] + n[1] * d2[2],
        c[2] + bc[2] * d2[0] + m[2] * d2[1] + n[2] * d2[2],
    )


def jacobi_eigh(matrix: Sequence[Sequence[float]], tol: float = 1e-24, max_sweeps: int = 100) -> tuple[list[float], list[list[float]]]:
    """Eigen-decomposition of a real symmetric matrix by cyclic Jacobi rotations.

    Returns eigenvalues in descending order and the eigenvectors as *columns*
    (``vectors[i][k]`` is component ``i`` of eigenvector ``k``). Iteration
    stops when the sum of squared off-diagonal elements falls below *tol*.
    """
    n = len(matrix)
    a = [list(map(float, row)) for row in matrix]
    if any(len(row) != n for row in a):
        raise ValueError("matrix must be square")
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: a[i][i], reverse=True)
    values = [a[i][i] for i in order]
    vectors = [[v[r][i] for i in order] for r in range(n)]
    return values, vectors
