"""Chemical reference data: residue alphabets and element radii.

Covalent radii follow Cordero et al. (2008), *Dalton Trans.* 2832-2838.
Van der Waals radii follow Bondi (1964), *J. Phys. Chem.* 68, 441, with the
customary 1.20 A value for hydrogen.
"""

from __future__ import annotations

from typing import Final

#: Three-letter to one-letter codes, including common modified and protonation variants.
THREE_TO_ONE: Final[dict[str, str]] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "UNK": "X", "MSE": "M", "HSD": "H", "HSE": "H",
    "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C",
    "ASX": "B", "GLX": "Z", "SEP": "S", "TPO": "T",
    "PTR": "Y", "MLY": "K", "CSO": "C", "HYP": "P",
}

#: Inverse of the canonical part of :data:`THREE_TO_ONE`.
ONE_TO_THREE: Final[dict[str, str]] = {v: k for k, v in list(THREE_TO_ONE.items())[:23]}

#: Common solvent residue names.
WATER_NAMES: Final[frozenset[str]] = frozenset({"HOH", "WAT", "H2O", "DOD", "SOL", "TIP", "TIP3"})

#: Backbone heavy-atom names for polypeptides.
BACKBONE_ATOMS: Final[tuple[str, ...]] = ("N", "CA", "C", "O")

#: Covalent radii in Angstrom.
COVALENT_RADII: Final[dict[str, float]] = {
    "H": 0.31, "D": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "P": 1.07, "SE": 1.20, "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "FE": 1.32, "ZN": 1.22, "MG": 1.41, "CA": 1.76, "NA": 1.66, "K": 2.03,
    "MN": 1.39, "CU": 1.32, "CO": 1.26, "NI": 1.24, "B": 0.84,
}

#: Van der Waals radii in Angstrom.
VDW_RADII: Final[dict[str, float]] = {
    "H": 1.20, "D": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "SE": 1.90, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 1.94, "ZN": 1.39, "MG": 1.73, "CA": 2.31, "NA": 2.27, "K": 2.75,
    "MN": 1.97, "CU": 1.40, "CO": 1.92, "NI": 1.63, "B": 1.92,
}

DEFAULT_COVALENT_RADIUS: Final[float] = 0.77
DEFAULT_VDW_RADIUS: Final[float] = 1.80

#: Two-letter elements recognised when inferring elements from atom-name columns 13-14.
TWO_LETTER_ELEMENTS: Final[frozenset[str]] = frozenset(
    {"CL", "BR", "FE", "ZN", "MG", "NA", "MN", "CU", "CO", "NI", "SE", "CA", "K"}
)


def covalent_radius(element: str) -> float:
    return COVALENT_RADII.get(element.upper(), DEFAULT_COVALENT_RADIUS)


def vdw_radius(element: str) -> float:
    return VDW_RADII.get(element.upper(), DEFAULT_VDW_RADIUS)


def one_letter(resname: str, unknown: str = "X") -> str:
    """Translate a three-letter residue name into its one-letter code."""
    return THREE_TO_ONE.get(resname.upper(), unknown)
