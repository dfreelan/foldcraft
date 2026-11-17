"""Distance matrices and residue contact maps.

A contact map doubles as the residue-level interaction graph. Contacts
default to the CASP convention (C-beta, or C-alpha for glycine,
closer than 8 A); ``mode="heavy"`` uses the minimum heavy-atom distance.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from . import geometry as geo
from .geometry import Vec3
from .spatial import CellGrid
from .structure import Chain, Model, Residue, Structure

#: Sequence-separation ranges used in contact evaluation (inclusive bounds).
SEPARATION_RANGES: dict[str, tuple[int, int | None]] = {"short": (6, 11), "medium": (12, 23), "long": (24, None)}


def distance_matrix(points: Sequence[Vec3]) -> list[list[float]]:
    """Symmetric matrix of pairwise Euclidean distances."""
    n = len(points)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mat[i][j] = mat[j][i] = geo.distance(points[i], points[j])
    return mat


def representative_atom(residue: Residue, mode: str = "CB") -> Vec3 | None:
    """Coordinate representing *residue*: ``"CA"``, or ``"CB"`` with a C-alpha fallback."""
    if mode not in ("CA", "CB"):
        raise ValueError(f"unknown representative mode {mode!r}")
    atom = residue.get("CB") if mode == "CB" else None
    atom = atom or residue.get("CA")
    return atom.coord if atom else None


@dataclass(frozen=True)
class ContactStats:
    """Precision / recall of a predicted contact set against a reference."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


class ContactMap:
    """Binary residue contact map over ``n`` residues; pairs are stored as ``(i, j)``, ``i < j``."""

    def __init__(
        self,
        n: int,
        contacts: Iterable[tuple[int, int]],
        labels: Sequence[str] | None = None,
        distances: dict[tuple[int, int], float] | None = None,
    ) -> None:
        self.n = n
        self.labels = list(labels) if labels is not None else [str(i + 1) for i in range(n)]
        if len(self.labels) != n:
            raise ValueError("labels must have length n")
        pairs = set()
        for i, j in contacts:
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise ValueError(f"invalid contact ({i}, {j}) for n={n}")
            pairs.add((min(i, j), max(i, j)))
        self.contacts: frozenset[tuple[int, int]] = frozenset(pairs)
        self.distances = dict(distances or {})

    @classmethod
    def from_structure(
        cls,
        obj: Structure | Model | Chain | Iterable[Residue],
        cutoff: float = 8.0,
        mode: str = "CB",
        min_separation: int = 1,
    ) -> ContactMap:
        """Build a map from coordinates (``mode`` is ``"CA"``, ``"CB"`` or ``"heavy"``)."""
        if isinstance(obj, Structure):
            obj = obj.model
        if isinstance(obj, Model):
            residues = [r for c in obj for r in c.polymer()]
        else:
            residues = obj.polymer() if isinstance(obj, Chain) else list(obj)
        labels = [f"{r.parent.id if r.parent else ''}:{r.id}" for r in residues]
        if mode == "heavy":
            sites = [(a.coord, k) for k, r in enumerate(residues) for a in r if not a.is_hydrogen]
        else:
            reps = [(representative_atom(r, mode), k) for k, r in enumerate(residues)]
            sites = [(p, k) for p, k in reps if p is not None]  # type: ignore[misc]
        found: dict[tuple[int, int], float] = {}
        if sites:
            grid: CellGrid[int] = CellGrid([p for p, _ in sites], cutoff)
            for i, j, d in grid.pairs(cutoff):
                a, b = sorted((sites[i][1], sites[j][1]))
                if b - a >= max(1, min_separation) and d < found.get((a, b), float("inf")):
                    found[(a, b)] = d
        return cls(len(residues), found, labels, found)

    def __contains__(self, pair: object) -> bool:
        if not isinstance(pair, tuple) or len(pair) != 2:
            return False
        return (min(pair), max(pair)) in self.contacts

    def __len__(self) -> int:
        return len(self.contacts)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(sorted(self.contacts))

    def filter(self, min_separation: int = 0, max_separation: int | None = None) -> ContactMap:
        """New map restricted to ``min_separation <= j - i <= max_separation``."""
        kept = {(i, j) for i, j in self.contacts
                if j - i >= min_separation and (max_separation is None or j - i <= max_separation)}
        return ContactMap(self.n, kept, self.labels, {k: v for k, v in self.distances.items() if k in kept})

    def by_range(self, name: str) -> ContactMap:
        """Contacts in the named range: ``short``, ``medium`` or ``long``."""
        return self.filter(*SEPARATION_RANGES[name])

    def neighbors(self, i: int) -> list[int]:
        """Residues in contact with residue *i* (the map viewed as a residue graph)."""
        return sorted(b if a == i else a for a, b in self.contacts if i in (a, b))

    def degrees(self) -> list[int]:
        """Number of contacts per residue."""
        deg = [0] * self.n
        for i, j in self.contacts:
            deg[i] += 1
            deg[j] += 1
        return deg

    def to_matrix(self) -> list[list[int]]:
        """Dense symmetric 0/1 matrix."""
        mat = [[0] * self.n for _ in range(self.n)]
        for i, j in self.contacts:
            mat[i][j] = mat[j][i] = 1
        return mat

    def relative_contact_order(self) -> float:
        """Relative contact order (Plaxco, Simons & Baker, 1998, *J. Mol. Biol.* 277, 985)."""
        if not self.contacts:
            return 0.0
        return sum(j - i for i, j in self.contacts) / (self.n * len(self.contacts))

    def compare(self, reference: ContactMap) -> ContactStats:
        """Score this (predicted) map against a *reference* map of equal size."""
        if reference.n != self.n:
            raise ValueError("contact maps differ in size")
        tp = len(self.contacts & reference.contacts)
        return ContactStats(tp, len(self.contacts) - tp, len(reference.contacts) - tp)

    def to_tsv(self) -> str:
        """Tab-separated listing with 1-based indices, labels and distances."""
        lines = ["i\tj\tres_i\tres_j\tdistance"]
        for i, j in self:
            d = self.distances.get((i, j))
            lines.append(f"{i + 1}\t{j + 1}\t{self.labels[i]}\t{self.labels[j]}\t{'' if d is None else f'{d:.3f}'}")
        return "\n".join(lines) + "\n"
