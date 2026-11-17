"""Ideal polypeptide backbones from torsion angles (Engh & Huber, 1991, *Acta Cryst.* A47, 392)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from . import geometry as geo
from .constants import ONE_TO_THREE
from .geometry import Vec3
from .structure import Atom, Chain, Model, Residue, Structure

# Engh & Huber ideal geometry (Angstrom / degrees).
N_CA = 1.458
CA_C = 1.525
C_N = 1.329
C_O = 1.231
CA_CB = 1.530
ANGLE_N_CA_C = 111.2
ANGLE_CA_C_N = 116.2
ANGLE_C_N_CA = 121.7
ANGLE_CA_C_O = 120.5
ANGLE_N_CA_CB = 110.5
DIHEDRAL_CB = -122.6  # improper C-N-CA-CB for L-amino acids

#: Canonical (phi, psi) pairs for common conformations.
CONFORMATIONS: dict[str, tuple[float, float]] = {
    "alpha": (-57.0, -47.0), "310": (-49.0, -26.0), "pi": (-57.0, -70.0), "beta": (-120.0, 130.0),
    "parallel_beta": (-119.0, 113.0), "antiparallel_beta": (-139.0, 135.0), "ppii": (-75.0, 145.0),
}


def build_backbone(
    torsions: Sequence[tuple[float, float]] | str,
    sequence: str | None = None,
    chain_id: str = "A",
    omega: float = 180.0,
    add_cb: bool = True,
    start: int = 1,
    length: int | None = None,
) -> Structure:
    """Build an ideal backbone (N, CA, C, O and, except for glycine, CB).

    *torsions* is a per-residue list of ``(phi, psi)`` in degrees or the name
    of an entry in :data:`CONFORMATIONS` (sized by *length* or *sequence*).
    *sequence* defaults to poly-alanine; residues are numbered from *start*.
    """
    if isinstance(torsions, str):
        if torsions not in CONFORMATIONS:
            raise ValueError(f"unknown conformation {torsions!r}")
        n = length if length is not None else (len(sequence) if sequence else None)
        if n is None:
            raise ValueError("length or sequence required with a named conformation")
        torsions = [CONFORMATIONS[torsions]] * n
    n = len(torsions)
    if n == 0:
        raise ValueError("at least one residue is required")
    seq = sequence if sequence is not None else "A" * n
    if len(seq) != n:
        raise ValueError("sequence length does not match the number of torsion pairs")

    n_xyz: Vec3 = (0.0, 0.0, 0.0)
    ca_xyz: Vec3 = (N_CA, 0.0, 0.0)
    theta = math.radians(180.0 - ANGLE_N_CA_C)
    c_xyz: Vec3 = (ca_xyz[0] + CA_C * math.cos(theta), CA_C * math.sin(theta), 0.0)
    prev_n, prev_ca, prev_c = n_xyz, ca_xyz, c_xyz

    structure = Structure("ideal")
    model = structure.add_model(Model(1))
    chain = model.add_chain(Chain(chain_id))
    serial = 1
    for i, (phi, psi) in enumerate(torsions):
        if i > 0:
            prev_psi = torsions[i - 1][1]
            n_xyz = geo.place_atom(prev_n, prev_ca, prev_c, C_N, ANGLE_CA_C_N, prev_psi)
            ca_xyz = geo.place_atom(prev_ca, prev_c, n_xyz, N_CA, ANGLE_C_N_CA, omega)
            c_xyz = geo.place_atom(prev_c, n_xyz, ca_xyz, CA_C, ANGLE_N_CA_C, phi)
        o_xyz = geo.place_atom(n_xyz, ca_xyz, c_xyz, C_O, ANGLE_CA_C_O, psi + 180.0)
        code = seq[i].upper()
        res = chain.add_residue(Residue(ONE_TO_THREE.get(code, "UNK"), start + i))
        for name, element, xyz in (("N", "N", n_xyz), ("CA", "C", ca_xyz), ("C", "C", c_xyz), ("O", "O", o_xyz)):
            res.add_atom(Atom(name, element, xyz, serial=serial))
            serial += 1
        if add_cb and code != "G":
            cb = geo.place_atom(c_xyz, n_xyz, ca_xyz, CA_CB, ANGLE_N_CA_CB, DIHEDRAL_CB)
            res.add_atom(Atom("CB", "C", cb, serial=serial))
            serial += 1
        prev_n, prev_ca, prev_c = n_xyz, ca_xyz, c_xyz
    return structure



def build_sheet(
    length: int = 8,
    orientation: str = "antiparallel",
    spacing: float = 5.0,
    chain_ids: tuple[str, str] = ("A", "B"),
) -> Structure:
    """Build an idealised two-stranded ``"parallel"`` or ``"antiparallel"`` sheet.

    The second strand is a copy of the first, rotated by 180 degrees about
    the carbonyl direction (antiparallel) or not (parallel), and displaced by
    *spacing* Angstrom along it, putting N-H and C=O groups in register.
    """
    if orientation not in ("parallel", "antiparallel"):
        raise ValueError("orientation must be 'parallel' or 'antiparallel'")
    if length < 4:
        raise ValueError("strands need at least four residues")
    conf = "parallel_beta" if orientation == "parallel" else "antiparallel_beta"
    sheet = build_backbone(conf, length=length, chain_id=chain_ids[0])
    partner = build_backbone(conf, length=length, chain_id=chain_ids[1])
    first = sheet.model[chain_ids[0]]
    ca = [r["CA"].coord for r in first]
    axis = geo.normalize(geo.sub(ca[-1], ca[0]))
    center = geo.centroid(ca)
    mid = first.residues[length // 2]
    carbonyl = geo.sub(mid["O"].coord, mid["C"].coord)
    perp = geo.normalize(geo.sub(carbonyl, geo.scale(axis, geo.dot(carbonyl, axis))))
    rotation = geo.rotation_about_axis(perp, 180.0) if orientation == "antiparallel" else geo.IDENTITY
    shift = geo.scale(perp, spacing)
    for atom in partner.all_atoms():
        local = geo.mat_vec(rotation, geo.sub(atom.coord, center))
        atom.coord = geo.add(geo.add(local, center), shift)
    second = partner.model.chains.pop(chain_ids[1])
    sheet.model.add_chain(second)
    sheet.id = f"ideal_{orientation}_sheet"
    return sheet
