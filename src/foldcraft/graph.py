"""Covalent topology: the atom-level bond graph and chain-break detection.

Intra-residue bonds are inferred from covalent radii (``d <= r_i + r_j +
tolerance``); inter-residue bonds are limited to peptide ``C-N``,
phosphodiester ``O3'-P`` and disulfide ``SG-SG`` links, so steric clashes are
never mistaken for covalent bonds.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from .constants import covalent_radius
from .structure import Atom, Chain, Model, Residue, Structure

PEPTIDE_BOND_MAX = 2.0
DISULFIDE_MAX = 2.5


def _chains(obj: Structure | Model | Chain) -> list[Chain]:
    if isinstance(obj, Chain):
        return [obj]
    return list(obj.model if isinstance(obj, Structure) else obj)


class MolecularGraph:
    """Undirected graph of atoms (nodes) and covalent bonds (edges)."""

    def __init__(self, atoms: list[Atom]) -> None:
        self.atoms = atoms
        self.adjacency: list[set[int]] = [set() for _ in atoms]
        self._index = {id(a): i for i, a in enumerate(atoms)}

    def index(self, atom: Atom) -> int:
        return self._index[id(atom)]

    def add_bond(self, i: int, j: int) -> None:
        if i == j:
            raise ValueError("self-bonds are not allowed")
        self.adjacency[i].add(j)
        self.adjacency[j].add(i)

    def has_bond(self, a: Atom, b: Atom) -> bool:
        return self.index(b) in self.adjacency[self.index(a)]

    def bonds(self) -> Iterator[tuple[int, int]]:
        """Each bond once, as ``(i, j)`` with ``i < j``."""
        for i, nbrs in enumerate(self.adjacency):
            yield from ((i, j) for j in sorted(nbrs) if i < j)

    @property
    def n_bonds(self) -> int:
        return sum(len(n) for n in self.adjacency) // 2

    def neighbors(self, atom: Atom) -> list[Atom]:
        return [self.atoms[j] for j in sorted(self.adjacency[self.index(atom)])]

    def within_bonds(self, i: int, max_bonds: int) -> set[int]:
        """Atoms reachable from atom *i* through at most *max_bonds* bonds."""
        depth = {i: 0}
        queue = deque([i])
        while queue:
            cur = queue.popleft()
            if depth[cur] < max_bonds:
                for nb in self.adjacency[cur] - depth.keys():
                    depth[nb] = depth[cur] + 1
                    queue.append(nb)
        return set(depth) - {i}

    def connected_components(self) -> list[list[int]]:
        """Components as sorted index lists, largest first."""
        seen: set[int] = set()
        comps = []
        for start in range(len(self.atoms)):
            if start not in seen:
                comp = self.within_bonds(start, len(self.atoms)) | {start}
                seen |= comp
                comps.append(sorted(comp))
        return sorted(comps, key=lambda c: (-len(c), c[0]))


def bond_graph(obj: Structure | Model | Chain, tolerance: float = 0.45) -> MolecularGraph:
    """Infer the covalent bond graph of a structure (first model), model or chain."""
    chains = _chains(obj)
    graph = MolecularGraph([a for c in chains for a in c.atoms()])
    idx = graph.index
    for chain in chains:
        prev: Residue | None = None
        for res in chain:
            atoms = list(res)
            for a, x in enumerate(atoms):
                for y in atoms[a + 1 :]:
                    if x.is_hydrogen and y.is_hydrogen:
                        continue
                    limit = covalent_radius(x.element) + covalent_radius(y.element) + tolerance
                    if 0.4 < x.distance_to(y) <= limit:
                        graph.add_bond(idx(x), idx(y))
            if prev is not None:
                for left, right in (("C", "N"), ("O3'", "P")):
                    p, q = prev.get(left), res.get(right)
                    if p is not None and q is not None and p.distance_to(q) <= PEPTIDE_BOND_MAX:
                        graph.add_bond(idx(p), idx(q))
            prev = res
    sulfurs = [a for a in graph.atoms if a.name == "SG" and a.element == "S"]
    for k, s1 in enumerate(sulfurs):
        for s2 in sulfurs[k + 1 :]:
            if s1.parent is not s2.parent and s1.distance_to(s2) <= DISULFIDE_MAX:
                graph.add_bond(idx(s1), idx(s2))
    return graph


@dataclass(frozen=True)
class ChainBreak:
    """Consecutive residues lacking a peptide bond (``distance`` is ``None`` if atoms are missing)."""

    chain_id: str
    before: Residue
    after: Residue
    distance: float | None


def find_chain_breaks(chain: Chain, max_peptide: float = PEPTIDE_BOND_MAX) -> list[ChainBreak]:
    """Report consecutive amino acids whose ``C(i)-N(i+1)`` link is missing or too long."""
    poly = chain.polymer()
    breaks = []
    for a, b in zip(poly, poly[1:]):
        c, n = a.get("C"), b.get("N")
        d = c.distance_to(n) if c is not None and n is not None else None
        if d is None or d > max_peptide:
            breaks.append(ChainBreak(chain.id, a, b, d))
    return breaks

