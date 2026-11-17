"""Residue correspondence between a model and a reference.

``numbering`` pairs residues with identical chain, sequence number and
insertion code; ``order`` pairs the k-th amino acid of each chain; and
``sequence`` uses a global Needleman-Wunsch pairing of the one-letter
sequences, which tolerates insertions, deletions and renumbering.
"""

from __future__ import annotations

from collections.abc import Sequence

from .errors import ComparisonError
from .geometry import Vec3
from .structure import Chain, Model, Residue, Structure

AlignedIndex = tuple[int | None, int | None]


def needleman_wunsch(a: str, b: str, match: float = 2.0, mismatch: float = -1.0,
                     gap: float = -2.0) -> tuple[list[AlignedIndex], float]:
    """Global Needleman-Wunsch pairing with a linear gap penalty.

    Returns ``(path, score)`` where *path* lists ``(i, j)`` index pairs and
    ``None`` marks a gap.
    """
    n, m = len(a), len(b)
    score = [[j * gap for j in range(m + 1)]] + [[i * gap] + [0.0] * m for i in range(1, n + 1)]
    trace = [[2] * (m + 1)] + [[1] + [0] * m for _ in range(n)]  # 0 diagonal, 1 up, 2 left
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                score[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch),
                score[i - 1][j] + gap,
                score[i][j - 1] + gap,
            )
            best = max(range(3), key=lambda k: (options[k], -k))
            score[i][j], trace[i][j] = options[best], best
    path: list[AlignedIndex] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = trace[i][j]
        path.append((i - 1 if move != 2 else None, j - 1 if move != 1 else None))
        i, j = i - (move != 2), j - (move != 1)
    return path[::-1], score[n][m]


def _chains(obj: Structure | Model | Chain) -> list[Chain]:
    if isinstance(obj, Chain):
        return [obj]
    return list(obj.model if isinstance(obj, Structure) else obj)


def pair_residues(model: Structure | Model | Chain, reference: Structure | Model | Chain,
                  method: str = "numbering") -> list[tuple[Residue, Residue]]:
    """Corresponding ``(model_residue, reference_residue)`` pairs.

    For ``order`` and ``sequence`` chains are matched by identifier when every
    model chain exists in the reference, otherwise positionally.
    """
    if method not in ("numbering", "order", "sequence"):
        raise ValueError(f"unknown pairing method {method!r}")
    mchains, rchains = _chains(model), _chains(reference)
    rmap = {c.id: c for c in rchains}
    if method == "numbering":
        return [(r, rmap[c.id][r.id]) for c in mchains if c.id in rmap for r in c.polymer() if r.id in rmap[c.id]]
    if all(c.id in rmap for c in mchains):
        chain_pairs = [(c, rmap[c.id]) for c in mchains]
    else:
        chain_pairs = list(zip(mchains, rchains))
    pairs: list[tuple[Residue, Residue]] = []
    for mc, rc in chain_pairs:
        mpoly, rpoly = mc.polymer(), rc.polymer()
        if method == "order":
            pairs.extend(zip(mpoly, rpoly))
        else:
            path, _ = needleman_wunsch(mc.sequence(), rc.sequence())
            pairs.extend((mpoly[i], rpoly[j]) for i, j in path if i is not None and j is not None)
    return pairs


def paired_coordinates(pairs: Sequence[tuple[Residue, Residue]],
                       atom_names: Sequence[str] = ("CA",)) -> tuple[list[Vec3], list[Vec3]]:
    """Matched coordinates for *atom_names*; atoms missing on either side are skipped."""
    mob: list[Vec3] = []
    ref: list[Vec3] = []
    for mres, rres in pairs:
        for name in atom_names:
            ma, ra = mres.get(name), rres.get(name)
            if ma is not None and ra is not None:
                mob.append(ma.coord)
                ref.append(ra.coord)
    if not mob:
        raise ComparisonError(f"no matching atoms named {list(atom_names)} in residue pairs")
    return mob, ref
