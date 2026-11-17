"""Steric clash detection.

Two atoms clash when closer than the sum of their van der Waals radii minus
an overlap tolerance (0.4 A, MolProbity's "serious clash" threshold). Pairs
within three covalent bonds are excluded, polar N/O pairs get an extra
hydrogen-bond allowance, and the clash score is clashes per 1000 atoms.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import VDW_RADII, vdw_radius
from .graph import bond_graph
from .spatial import CellGrid
from .structure import Atom, Chain, Model, Structure

_POLAR = frozenset({"N", "O"})


@dataclass(frozen=True)
class Clash:
    """A pair of atoms closer than allowed by their van der Waals radii."""

    atom_a: Atom
    atom_b: Atom
    distance: float
    overlap: float

    def describe(self) -> str:
        return f"{self.atom_a.label()} -- {self.atom_b.label()}  d={self.distance:.2f} A  overlap={self.overlap:.2f} A"


@dataclass(frozen=True)
class ClashReport:
    """All clashes in a model with summary statistics."""

    clashes: list[Clash]
    n_atoms: int

    @property
    def clash_score(self) -> float:
        """Clashes per 1000 atoms."""
        return 1000.0 * len(self.clashes) / self.n_atoms if self.n_atoms else 0.0

    @property
    def worst(self) -> Clash | None:
        return max(self.clashes, key=lambda c: c.overlap, default=None)

    def __len__(self) -> int:
        return len(self.clashes)


def find_clashes(
    obj: Structure | Model | Chain,
    tolerance: float = 0.4,
    include_hydrogens: bool = False,
    hbond_allowance: float = 0.5,
    bond_separation: int = 3,
    radii: dict[str, float] | None = None,
) -> ClashReport:
    """Detect steric clashes in a structure (first model), model or chain.

    *tolerance* is the permitted overlap in Angstrom, *hbond_allowance* the
    extra overlap allowed for N/O pairs, and atom pairs within
    *bond_separation* covalent bonds are ignored. *radii* overrides entries of
    :data:`~foldcraft.constants.VDW_RADII`.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    table = dict(VDW_RADII)
    if radii:
        table.update({k.upper(): v for k, v in radii.items()})
    graph = bond_graph(obj)
    selected = [i for i, a in enumerate(graph.atoms) if include_hydrogens or not a.is_hydrogen]
    if not selected:
        return ClashReport([], 0)
    atoms = [graph.atoms[i] for i in selected]
    rad = [table.get(a.element, vdw_radius(a.element)) for a in atoms]
    cutoff = 2.0 * max(rad)
    grid: CellGrid[int] = CellGrid([a.coord for a in atoms], cutoff)
    exclusion_cache: dict[int, set[int]] = {}
    clashes: list[Clash] = []
    for i, j, d in grid.pairs(cutoff):
        a, b = atoms[i], atoms[j]
        allowed = rad[i] + rad[j] - tolerance
        if a.element in _POLAR and b.element in _POLAR:
            allowed -= hbond_allowance
        if d >= allowed:
            continue
        gi, gj = selected[i], selected[j]
        if gi not in exclusion_cache:
            exclusion_cache[gi] = graph.within_bonds(gi, bond_separation)
        if gj in exclusion_cache[gi]:
            continue
        if a.altloc and b.altloc and a.altloc != b.altloc and a.parent is b.parent:
            continue
        clashes.append(Clash(a, b, d, rad[i] + rad[j] - d))
    clashes.sort(key=lambda c: -c.overlap)
    return ClashReport(clashes, len(atoms))


def clash_score(obj: Structure | Model | Chain, **options: object) -> float:
    """Convenience wrapper returning only the clash score."""
    return find_clashes(obj, **options).clash_score  # type: ignore[arg-type]

