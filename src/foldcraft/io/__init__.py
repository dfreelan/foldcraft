"""Coordinate file input and output.

:func:`load` and :func:`save` dispatch on the file extension (falling back to
content sniffing on input) and handle gzip-compressed input transparently.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..structure import Structure
from ._source import sniff_format
from .builder import StructureBuilder
from .mmcif import CIFBlock, parse_cif_blocks, parse_mmcif_string, read_mmcif, structure_to_mmcif, tokenize, write_mmcif
from .pdb import PDBParser, hy36decode, hy36encode, infer_element, parse_pdb_string, read_pdb, to_pdb_string, write_pdb

__all__ = [
    "CIFBlock", "PDBParser", "StructureBuilder", "hy36decode", "hy36encode", "infer_element", "load",
    "parse_cif_blocks", "parse_mmcif_string", "parse_pdb_string", "read_mmcif", "read_pdb", "save",
    "sniff_format", "structure_to_mmcif", "to_pdb_string", "tokenize", "write_mmcif", "write_pdb",
]


def _kind(fmt: str) -> str:
    kind = fmt.lower()
    if kind not in ("pdb", "mmcif", "cif"):
        raise ValueError(f"unsupported structure format {fmt!r}")
    return "mmcif" if kind == "cif" else kind


def load(path: str | os.PathLike[str], fmt: str | None = None, **options: object) -> Structure:
    """Load a structure; *options* (``altloc``, ``strict`` ...) go to the format reader."""
    reader = read_pdb if _kind(fmt or sniff_format(path)) == "pdb" else read_mmcif
    return reader(path, **options)  # type: ignore[operator]


def save(structure: Structure, path: str | os.PathLike[str], fmt: str | None = None) -> None:
    """Write *structure* as PDB or mmCIF, chosen by *fmt* or the file extension."""
    kind = _kind(fmt) if fmt else ("mmcif" if Path(path).suffix.lower() in (".cif", ".mmcif") else "pdb")
    (write_pdb if kind == "pdb" else write_mmcif)(structure, path)
