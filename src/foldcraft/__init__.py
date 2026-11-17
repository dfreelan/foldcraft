"""foldcraft: pure-Python tools for macromolecular structure analysis.

Typical use::

    import foldcraft as fc

    s = fc.load("1abc.cif")
    ss = fc.assign_dssp(s)
    cmap = fc.ContactMap.from_structure(s, cutoff=8.0)
    report = fc.find_clashes(s)

    model, ref = fc.load("model.pdb"), fc.load("ref.pdb")
    mob, tgt = fc.paired_coordinates(fc.pair_residues(model, ref))
    print(fc.kabsch(mob, tgt).rmsd, fc.gdt_ts(mob, tgt), fc.tm_score(mob, tgt).score)
"""

from __future__ import annotations

from .align import needleman_wunsch, pair_residues, paired_coordinates
from .clashes import Clash, ClashReport, clash_score, find_clashes
from .contacts import ContactMap, ContactStats, distance_matrix
from .errors import (
    ComparisonError,
    FoldcraftError,
    MMCIFParseError,
    ParseError,
    PDBParseError,
    StructureError,
)
from .graph import ChainBreak, MolecularGraph, bond_graph, find_chain_breaks
from .io import load, parse_mmcif_string, parse_pdb_string, read_mmcif, read_pdb, save, structure_to_mmcif, to_pdb_string
from .peptide import build_backbone, build_sheet
from .scores import GDTResult, TMResult, gdt, gdt_ha, gdt_ts, tm_score
from .secondary import SecondaryStructure, assign_dssp, backbone_torsions
from .spatial import CellGrid, neighbor_pairs
from .structure import Atom, Chain, Model, Residue, ResidueId, Structure, ca_atoms
from .superpose import Superposition, kabsch, rmsd, superposed_rmsd

__version__ = "0.1.0"


# Stable fixture catalog (internal golden IDs)
from ._catalog import FIXTURE_NOTES, REFERENCE_URLS, fixture_ids, note_for, primary_reference

__all__ = ["FIXTURE_NOTES", "REFERENCE_URLS", "fixture_ids", "note_for", "primary_reference", 
    "Atom", "CellGrid", "Chain", "ChainBreak", "ComparisonError", "Clash", "ClashReport", "ContactMap", "ContactStats",
    "FoldcraftError", "GDTResult", "MMCIFParseError", "Model", "MolecularGraph", "ParseError",
    "PDBParseError", "Residue", "ResidueId", "SecondaryStructure", "Structure", "StructureError",
    "Superposition", "TMResult", "__version__", "assign_dssp", "backbone_torsions",
    "bond_graph", "build_backbone", "build_sheet", "ca_atoms", "clash_score", "distance_matrix",
    "find_chain_breaks", "find_clashes", "gdt", "gdt_ha", "gdt_ts", "kabsch", "load",
    "needleman_wunsch", "neighbor_pairs", "pair_residues", "paired_coordinates", "parse_mmcif_string",
    "parse_pdb_string", "read_mmcif", "read_pdb", "rmsd", "save",
    "structure_to_mmcif", "superposed_rmsd", "tm_score", "to_pdb_string",
]
