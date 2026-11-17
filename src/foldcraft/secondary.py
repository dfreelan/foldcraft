"""DSSP-style secondary-structure assignment.

Follows Kabsch & Sander (1983), *Biopolymers* 22, 2577: backbone H-bonds have
energy ``E = 0.084 * 332 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)`` kcal/mol
(bonded below -0.5) with ``H = N + unit(C_prev - O_prev)``. Helices
(``H``/``G``/``I``) come from consecutive n-turns, ladders (``E``) and
bridges (``B``) from bridge patterns, then turns (``T``) and bends (``S``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import geometry as geo
from .geometry import Vec3
from .spatial import CellGrid
from .structure import Chain, Model, Residue, Structure

HBOND_ENERGY_CUTOFF = -0.5
HBOND_ENERGY_FLOOR = -9.9
_Q1Q2F = 0.084 * 332.0
_PEPTIDE_MAX = 2.5
_CA_SEARCH = 9.0


@dataclass
class SecondaryStructure:
    """Per-residue codes plus the hydrogen bonds and bridges behind them.

    ``hbonds`` maps ``(acceptor_CO_index, donor_NH_index)`` to the energy;
    ``bridges`` lists ``(i, j, "P" | "A")`` for parallel / antiparallel bridges.
    """

    residues: list[Residue]
    codes: list[str]
    hbonds: dict[tuple[int, int], float] = field(default_factory=dict)
    bridges: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def string(self) -> str:
        return "".join(self.codes)

    def __len__(self) -> int:
        return len(self.codes)

    def segments(self) -> list[tuple[str, int, int]]:
        """Runs of identical codes as ``(code, first_index, last_index)``."""
        out: list[tuple[str, int, int]] = []
        for i, c in enumerate(self.codes):
            if out and out[-1][0] == c:
                out[-1] = (c, out[-1][1], i)
            else:
                out.append((c, i, i))
        return out

    def three_state(self) -> str:
        """Reduce to the conventional H (H/G/I), E (E/B), C alphabet."""
        return "".join("H" if c in "HGI" else "E" if c in "EB" else "C" for c in self.codes)

    def composition(self) -> dict[str, float]:
        """Fractions of helix, strand and coil in the three-state reduction."""
        reduced = self.three_state()
        n = len(reduced) or 1
        return {"helix": reduced.count("H") / n, "strand": reduced.count("E") / n, "coil": reduced.count("C") / n}


def hbond_energy(n: Vec3, h: Vec3, c: Vec3, o: Vec3) -> float:
    """Kabsch-Sander electrostatic energy (kcal/mol) of an N-H ... O=C pair."""
    r_on, r_ch = geo.distance(o, n), geo.distance(c, h)
    r_oh, r_cn = geo.distance(o, h), geo.distance(c, n)
    if min(r_on, r_ch, r_oh, r_cn) < 1e-6:
        return HBOND_ENERGY_FLOOR
    return max(HBOND_ENERGY_FLOOR, _Q1Q2F * (1 / r_on + 1 / r_ch - 1 / r_oh - 1 / r_cn))


def _backbone_residues(obj: Structure | Model | Chain | Sequence[Residue]) -> list[Residue]:
    if isinstance(obj, (Structure, Model)):
        candidates = list(obj.residues())
    else:
        candidates = list(obj)
    return [r for r in candidates if r.has_backbone()]


def _links(res: list[Residue]) -> list[bool]:
    """``link[i]`` is True when residue *i* is peptide-bonded to residue *i-1*."""
    return [False] + [
        a.parent is b.parent and a["C"].distance_to(b["N"]) <= _PEPTIDE_MAX for a, b in zip(res, res[1:])
    ]


def backbone_hbonds(residues: Sequence[Residue]) -> dict[tuple[int, int], float]:
    """Backbone hydrogen bonds as ``{(acceptor_CO_index, donor_NH_index): energy}``."""
    res = list(residues)
    link = _links(res)
    hs: list[Vec3 | None] = [None] * len(res)
    for i, r in enumerate(res):
        if link[i] and r.name != "PRO":
            prev = res[i - 1]
            hs[i] = geo.add(r["N"].coord, geo.normalize(geo.sub(prev["C"].coord, prev["O"].coord)))
    grid: CellGrid[int] = CellGrid([r["CA"].coord for r in res], _CA_SEARCH)
    bonds: dict[tuple[int, int], float] = {}
    for a, b, _ in grid.pairs(_CA_SEARCH):
        for donor, acceptor in ((a, b), (b, a)):
            h = hs[donor]
            if h is None or (abs(donor - acceptor) < 2 and res[donor].parent is res[acceptor].parent):
                continue
            e = hbond_energy(res[donor]["N"].coord, h, res[acceptor]["C"].coord, res[acceptor]["O"].coord)
            if e < HBOND_ENERGY_CUTOFF:
                bonds[(acceptor, donor)] = e
    return bonds


def assign_dssp(obj: Structure | Model | Chain | Sequence[Residue]) -> SecondaryStructure:
    """Assign secondary structure to residues of *obj* that have N, CA, C and O.

    Priority follows DSSP: H > B > E > G > I > T > S.
    """
    res = _backbone_residues(obj)
    n = len(res)
    codes = ["-"] * n
    if n == 0:
        return SecondaryStructure(res, codes)
    link = _links(res)
    hb = backbone_hbonds(res)

    def chained(i: int, j: int) -> bool:
        return 0 <= i <= j < n and all(link[i + 1 : j + 1])

    def has(i: int, j: int) -> bool:
        return (i, j) in hb

    turns = {k: [chained(i, i + k) and has(i, i + k) for i in range(n)] for k in (3, 4, 5)}

    for i in range(1, n - 3):
        if turns[4][i - 1] and turns[4][i]:
            codes[i : i + 4] = ["H"] * 4

    bridges: list[tuple[int, int, str]] = []
    for i in range(1, n - 1):
        if not chained(i - 1, i + 1):
            continue
        for j in range(i + 3, n - 1):
            if not chained(j - 1, j + 1):
                continue
            if (has(i - 1, j) and has(j, i + 1)) or (has(j - 1, i) and has(i, j + 1)):
                bridges.append((i, j, "P"))
            elif (has(i, j) and has(j, i)) or (has(i - 1, j + 1) and has(j - 1, i + 1)):
                bridges.append((i, j, "A"))

    in_ladder: set[int] = set()
    for ladder in _ladders(bridges):
        if len(ladder) > 1:
            for side in (0, 1):
                ends = [b[side] for b in ladder]
                in_ladder.update(range(min(ends), max(ends) + 1))
    for i, j, _ in bridges:
        for k in (i, j):
            if codes[k] == "-" and k not in in_ladder:
                codes[k] = "B"
    for k in in_ladder:
        if codes[k] in "-B":
            codes[k] = "E"

    for size, code in ((3, "G"), (5, "I")):
        for i in range(1, n - size + 1):
            if turns[size][i - 1] and turns[size][i] and all(c in ("-", code) for c in codes[i : i + size]):
                codes[i : i + size] = [code] * size

    for size in (3, 4, 5):
        for i in range(n):
            if turns[size][i]:
                for k in range(i + 1, min(n, i + size)):
                    if codes[k] == "-":
                        codes[k] = "T"

    for i in range(2, n - 2):
        if codes[i] == "-" and chained(i - 2, i + 2):
            a, b, c = (res[k]["CA"].coord for k in (i - 2, i, i + 2))
            u, v = geo.sub(b, a), geo.sub(c, b)
            cos = geo.dot(u, v) / (geo.norm(u) * geo.norm(v))
            if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > 70.0:
                codes[i] = "S"
    return SecondaryStructure(res, codes, hb, bridges)


def _ladders(bridges: list[tuple[int, int, str]]) -> list[list[tuple[int, int, str]]]:
    """Group bridges into ladders, allowing beta bulges (gaps up to 1 and 4)."""
    ladders: list[list[tuple[int, int, str]]] = []
    for b in sorted(bridges):
        i, j, t = b
        for ladder in ladders:
            li, lj, lt = ladder[-1]
            di, dj = i - li, (j - lj) if t == "P" else (lj - j)
            if lt == t and di >= 1 and dj >= 1 and ((di <= 2 and dj <= 5) or (di <= 5 and dj <= 2)):
                ladder.append(b)
                break
        else:
            ladders.append([b])
    return ladders


@dataclass(frozen=True)
class BackboneTorsions:
    """Backbone dihedrals of one residue in degrees (``None`` at termini and breaks)."""

    residue: Residue
    phi: float | None
    psi: float | None
    omega: float | None


def backbone_torsions(chain: Chain) -> list[BackboneTorsions]:
    """Phi, psi and omega for each amino-acid residue of *chain*."""
    poly = [r for r in chain.polymer() if all(n in r.atoms for n in ("N", "CA", "C"))]
    out = []
    for k, r in enumerate(poly):
        prev = poly[k - 1] if k > 0 else None
        nxt = poly[k + 1] if k + 1 < len(poly) else None
        if prev is not None and prev["C"].distance_to(r["N"]) > _PEPTIDE_MAX:
            prev = None
        if nxt is not None and r["C"].distance_to(nxt["N"]) > _PEPTIDE_MAX:
            nxt = None
        n, ca, c = r["N"].coord, r["CA"].coord, r["C"].coord
        phi = geo.dihedral(prev["C"].coord, n, ca, c) if prev else None
        psi = geo.dihedral(n, ca, c, nxt["N"].coord) if nxt else None
        omega = geo.dihedral(ca, c, nxt["N"].coord, nxt["CA"].coord) if nxt else None
        out.append(BackboneTorsions(r, phi, psi, omega))
    return out
