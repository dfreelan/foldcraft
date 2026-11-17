"""Uniform-grid (cell list) neighbour search.

Cubic cells of edge ``r`` confine all pairs within ``r`` to adjacent cells,
making neighbour search roughly O(N) at macromolecular densities.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

from .geometry import Vec3

T = TypeVar("T")

_NEIGHBOUR_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


class CellGrid(Generic[T]):
    """Spatial hash over a fixed set of points with optional per-point payloads.

    Pair enumeration is exact for radii up to *cell_size*; :meth:`query`
    supports any radius (larger radii simply visit more cells).
    """

    def __init__(self, points: Sequence[Vec3], cell_size: float, items: Sequence[T] | None = None) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        if items is not None and len(items) != len(points):
            raise ValueError("items and points differ in length")
        self.points = list(points)
        self.items: list[T] = list(items) if items is not None else list(range(len(points)))  # type: ignore[arg-type]
        self.cell_size = float(cell_size)
        self._cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, p in enumerate(self.points):
            self._cells[self._cell(p)].append(i)

    def _cell(self, p: Vec3) -> tuple[int, int, int]:
        s = self.cell_size
        return (math.floor(p[0] / s), math.floor(p[1] / s), math.floor(p[2] / s))

    def __len__(self) -> int:
        return len(self.points)

    def query(self, center: Vec3, radius: float) -> list[tuple[T, float]]:
        """Return ``(item, distance)`` for every indexed point within *radius* of *center*."""
        r2 = radius * radius
        reach = max(1, math.ceil(radius / self.cell_size))
        cx, cy, cz = self._cell(center)
        found: list[tuple[T, float]] = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    for i in self._cells.get((cx + dx, cy + dy, cz + dz), ()):
                        p = self.points[i]
                        ddx, ddy, ddz = p[0] - center[0], p[1] - center[1], p[2] - center[2]
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz
                        if d2 <= r2:
                            found.append((self.items[i], math.sqrt(d2)))
        return found

    def pairs(self, radius: float | None = None) -> Iterator[tuple[int, int, float]]:
        """Yield ``(i, j, d)`` with ``i < j`` and ``d <= radius`` (default: the cell size)."""
        r = self.cell_size if radius is None else radius
        if r > self.cell_size + 1e-12:
            raise ValueError("pair radius may not exceed the cell size")
        r2 = r * r
        pts = self.points
        for (cx, cy, cz), members in self._cells.items():
            for dx, dy, dz in _NEIGHBOUR_OFFSETS:
                other = self._cells.get((cx + dx, cy + dy, cz + dz))
                if other is None:
                    continue
                for i in members:
                    pi = pts[i]
                    for j in other:
                        if j <= i:
                            continue
                        pj = pts[j]
                        ddx, ddy, ddz = pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz
                        if d2 <= r2:
                            yield i, j, math.sqrt(d2)


def neighbor_pairs(points: Sequence[Vec3], cutoff: float) -> list[tuple[int, int, float]]:
    """All index pairs ``(i, j, d)`` with ``i < j`` closer than *cutoff*, sorted."""
    if not points:
        return []
    grid: CellGrid[int] = CellGrid(points, cutoff)
    return sorted(grid.pairs(cutoff))
