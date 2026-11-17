"""Model-quality scores against a reference structure.

* GDT_TS / GDT_HA: Global Distance Test (Zemla, 2003, *Nucleic Acids Res.*
  31, 3370), the largest fraction of residues superposable under cutoffs of
  1, 2, 4, 8 A (TS) or 0.5, 1, 2, 4 A (HA).
* TM-score: Zhang & Skolnick (2004), *Proteins* 57, 702, with
  ``d0 = 1.24 (L - 15)^(1/3) - 1.8``.

Superposition-dependent scores use a maximal-subset search: superpositions
are seeded on contiguous fragments of several lengths and refined on the
residues under the distance threshold, following the strategy of the
reference programs (values can differ slightly from them).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from . import geometry as geo
from .errors import ComparisonError
from .geometry import Vec3
from .superpose import Superposition, kabsch

GDT_TS_CUTOFFS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
GDT_HA_CUTOFFS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


def _validate(model: Sequence[Vec3], reference: Sequence[Vec3]) -> None:
    if len(model) != len(reference):
        raise ComparisonError(f"model and reference differ in length ({len(model)} vs {len(reference)})")
    if len(model) < 3:
        raise ComparisonError("at least three corresponding points are required")


def _distances(sup: Superposition, model: Sequence[Vec3], reference: Sequence[Vec3]) -> list[float]:
    return [geo.distance(p, q) for p, q in zip(sup.apply_all(model), reference)]


def maximal_subset_search(
    model: Sequence[Vec3],
    reference: Sequence[Vec3],
    select_cutoff: float,
    objective: Callable[[list[float]], float],
    seed_lengths: Sequence[int] = (4, 8, 16),
    max_iterations: int = 20,
) -> tuple[float, Superposition]:
    """Return ``(best_value, superposition)`` maximising *objective* over per-point distances.

    Each seed fragment is superposed, then refined on the points closer than
    *select_cutoff* until that subset stops changing.
    """
    _validate(model, reference)
    n = len(model)
    best: tuple[float, Superposition | None] = (-math.inf, None)
    for length in sorted({min(max(3, L), n) for L in (*seed_lengths, n)}, reverse=True):
        for start in range(0, n - length + 1, max(1, length // 2)):
            subset, seen = list(range(start, start + length)), set()
            for _ in range(max_iterations):
                sup = kabsch([model[i] for i in subset], [reference[i] for i in subset])
                dists = _distances(sup, model, reference)
                value = objective(dists)
                if value > best[0]:
                    best = (value, sup)
                subset = [i for i, d in enumerate(dists) if d <= select_cutoff]
                if len(subset) < 3 or tuple(subset) in seen:
                    break
                seen.add(tuple(subset))
    assert best[1] is not None
    return best[0], best[1]


@dataclass(frozen=True)
class GDTResult:
    """GDT score and the per-cutoff fractions it averages."""

    score: float
    fractions: dict[float, float]


def gdt(model: Sequence[Vec3], reference: Sequence[Vec3], cutoffs: Sequence[float] = GDT_TS_CUTOFFS,
        n_reference: int | None = None) -> GDTResult:
    """Global Distance Test of corresponding C-alpha coordinates.

    Fractions are normalised by *n_reference* (default: the number of pairs);
    pass the full reference length when the model is partial.
    """
    _validate(model, reference)
    norm = n_reference if n_reference is not None else len(reference)
    if norm < len(reference):
        raise ValueError("n_reference cannot be smaller than the number of pairs")
    fractions = {
        c: maximal_subset_search(model, reference, c, lambda d, c=c: float(sum(x <= c for x in d)))[0] / norm
        for c in cutoffs
    }
    return GDTResult(sum(fractions.values()) / len(fractions), fractions)


def gdt_ts(model: Sequence[Vec3], reference: Sequence[Vec3], n_reference: int | None = None) -> float:
    """GDT_TS in ``[0, 1]`` (multiply by 100 for the conventional percentage)."""
    return gdt(model, reference, GDT_TS_CUTOFFS, n_reference).score


def gdt_ha(model: Sequence[Vec3], reference: Sequence[Vec3], n_reference: int | None = None) -> float:
    """High-accuracy GDT in ``[0, 1]``."""
    return gdt(model, reference, GDT_HA_CUTOFFS, n_reference).score


def tm_d0(length: int) -> float:
    """TM-score distance scale for a reference of *length* residues."""
    return 0.5 if length <= 21 else max(0.5, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)


@dataclass(frozen=True)
class TMResult:
    """TM-score, its distance scale, the optimal superposition and residues within 5 A."""

    score: float
    d0: float
    superposition: Superposition
    aligned_within_5: int


def tm_score(model: Sequence[Vec3], reference: Sequence[Vec3], n_reference: int | None = None) -> TMResult:
    """TM-score of *model* against *reference* for a fixed residue correspondence."""
    _validate(model, reference)
    length = n_reference if n_reference is not None else len(reference)
    d0 = tm_d0(length)
    score, sup = maximal_subset_search(
        model, reference, min(8.0, max(4.5, d0)),
        lambda dists: sum(1.0 / (1.0 + (d / d0) ** 2) for d in dists) / length,
    )
    return TMResult(score, d0, sup, sum(d <= 5.0 for d in _distances(sup, model, reference)))

