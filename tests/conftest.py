"""Shared fixtures: synthetic coordinate files and idealised structures."""

from pathlib import Path

import pytest

import foldcraft as fc

DATA = Path(__file__).parent / "data"


@pytest.fixture
def mini_pdb() -> Path:
    return DATA / "mini.pdb"


@pytest.fixture
def nmr_pdb() -> Path:
    return DATA / "nmr.pdb"


@pytest.fixture
def mini_cif() -> Path:
    return DATA / "mini.cif"


@pytest.fixture
def helix() -> fc.Structure:
    return fc.build_backbone("alpha", length=16)


@pytest.fixture
def strand() -> fc.Structure:
    return fc.build_backbone("beta", length=10)


@pytest.fixture
def antiparallel_sheet() -> fc.Structure:
    return fc.build_sheet(8, "antiparallel")


def atom_line(serial, name, resname, chain, seq, xyz, element="C", record="ATOM", altloc=" ", occupancy=1.0):
    """Format a PDB coordinate record for hand-built test inputs."""
    x, y, z = xyz
    return (f"{record:<6}{serial:>5}  {name:<3}{altloc}{resname:>3} {chain}{seq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{0.0:6.2f}          {element:>2}")
