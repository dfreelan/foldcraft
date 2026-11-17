"""Command-line interface: ``foldcraft {info,ss,contacts,clashes,compare,convert} ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from . import __version__
from .align import pair_residues, paired_coordinates
from .clashes import find_clashes
from .contacts import ContactMap
from .errors import FoldcraftError
from .graph import find_chain_breaks
from .io import load, save
from .scores import GDT_HA_CUTOFFS, GDT_TS_CUTOFFS, gdt, tm_score
from .secondary import assign_dssp
from .structure import Structure
from .superpose import kabsch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foldcraft", description="Protein structure parsing and analysis.")
    parser.add_argument("--version", action="version", version=f"foldcraft {__version__}")
    parser.add_argument("--lenient", action="store_true", help="skip malformed records instead of failing")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info", help="summarise models, chains, sequences and metadata")
    ss = sub.add_parser("ss", help="DSSP-style secondary structure per chain")
    contacts = sub.add_parser("contacts", help="residue contact list")
    clashes = sub.add_parser("clashes", help="steric clash report")
    for p in (info, ss, contacts, clashes):
        p.add_argument("file", help="PDB or mmCIF file (optionally gzip-compressed)")
    info.add_argument("--json", action="store_true")
    ss.add_argument("--three-state", action="store_true", help="reduce to H/E/C")
    contacts.add_argument("--cutoff", type=float, default=8.0)
    contacts.add_argument("--mode", choices=["CA", "CB", "heavy"], default="CB")
    contacts.add_argument("--min-sep", type=int, default=6, help="minimum sequence separation")
    contacts.add_argument("--summary", action="store_true", help="print counts instead of the list")
    clashes.add_argument("--tolerance", type=float, default=0.4)
    clashes.add_argument("--hydrogens", action="store_true")
    clashes.add_argument("--limit", type=int, default=20, help="number of clashes to list")
    compare = sub.add_parser("compare", help="RMSD, GDT and TM-score against a reference")
    compare.add_argument("model")
    compare.add_argument("reference")
    compare.add_argument("--pairing", choices=["numbering", "order", "sequence"], default="sequence")
    compare.add_argument("--json", action="store_true")
    convert = sub.add_parser("convert", help="convert between PDB and mmCIF (format from extension)")
    convert.add_argument("input")
    convert.add_argument("output")
    return parser


def _emit(data: dict[str, object], as_json: bool, out: TextIO) -> None:
    if as_json:
        out.write(json.dumps(data, indent=2, default=list) + "\n")
    else:
        out.writelines(f"{k}: {v}\n" for k, v in data.items())


def _info(s: Structure, args: argparse.Namespace, out: TextIO) -> None:
    chains = [{
        "id": c.id,
        "residues": len(c),
        "sequence": c.sequence(),
        "ligands": sorted({r.name for r in c if r.hetero and not r.is_water}),
        "waters": sum(1 for r in c if r.is_water),
        "breaks": len(find_chain_breaks(c)),
    } for c in s.model]
    meta = {k: s.metadata[k] for k in ("title", "method", "resolution", "space_group", "cell") if k in s.metadata}
    if args.json:
        _emit({"id": s.id, **s.summary(), "chains": chains, "metadata": meta}, True, out)
        return
    _emit({"id": s.id, **s.summary(), **meta}, False, out)
    for c in chains:
        ligands = f", ligands {','.join(c['ligands'])}" if c["ligands"] else ""
        out.write(f"chain {c['id']}: {c['residues']} residues, {c['breaks']} breaks, {c['waters']} waters{ligands}\n")
        out.write(f"  {c['sequence']}\n")
    for warning in s.metadata.get("warnings", []):  # type: ignore[attr-defined]
        out.write(f"warning: {warning}\n")


def _ss(s: Structure, args: argparse.Namespace, out: TextIO) -> None:
    for chain in s.model:
        result = assign_dssp(chain)
        if len(result):
            comp = result.composition()
            text = result.three_state() if args.three_state else result.string
            out.write(f">{s.id}:{chain.id} helix={comp['helix']:.2f} strand={comp['strand']:.2f}\n{text}\n")


def _contacts(s: Structure, args: argparse.Namespace, out: TextIO) -> None:
    cmap = ContactMap.from_structure(s, cutoff=args.cutoff, mode=args.mode, min_separation=args.min_sep)
    if not args.summary:
        out.write(cmap.to_tsv())
        return
    counts = {name: len(cmap.by_range(name)) for name in ("short", "medium", "long")}
    _emit({"residues": cmap.n, "contacts": len(cmap), **counts,
           "relative_contact_order": round(cmap.relative_contact_order(), 4)}, False, out)


def _clashes(s: Structure, args: argparse.Namespace, out: TextIO) -> None:
    report = find_clashes(s, tolerance=args.tolerance, include_hydrogens=args.hydrogens)
    _emit({"atoms": report.n_atoms, "clashes": len(report), "clash_score": round(report.clash_score, 2)}, False, out)
    out.writelines(f"  {c.describe()}\n" for c in report.clashes[: args.limit])


def _compare(model: Structure, args: argparse.Namespace, out: TextIO) -> None:
    ref = load(args.reference, strict=not args.lenient)
    mob, tgt = paired_coordinates(pair_residues(model, ref, method=args.pairing), ("CA",))
    n_ref = sum(1 for c in ref.model for r in c.polymer() if "CA" in r)
    _emit({
        "pairs": len(mob),
        "reference_length": n_ref,
        "rmsd": round(kabsch(mob, tgt).rmsd, 4),
        "gdt_ts": round(gdt(mob, tgt, GDT_TS_CUTOFFS, n_ref).score, 4),
        "gdt_ha": round(gdt(mob, tgt, GDT_HA_CUTOFFS, n_ref).score, 4),
        "tm_score": round(tm_score(mob, tgt, n_ref).score, 4),
    }, args.json, out)


def _convert(s: Structure, args: argparse.Namespace, out: TextIO) -> None:
    save(s, args.output)
    out.write(f"wrote {args.output}\n")


_COMMANDS = {"info": _info, "ss": _ss, "contacts": _contacts, "clashes": _clashes, "compare": _compare,
             "convert": _convert}


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Run the CLI and return the process exit status."""
    args = build_parser().parse_args(argv)
    path = getattr(args, "file", None) or getattr(args, "model", None) or args.input
    try:
        structure = load(path, strict=not args.lenient)
        _COMMANDS[args.command](structure, args, out or sys.stdout)
    except (FoldcraftError, OSError, ValueError) as exc:
        sys.stderr.write(f"foldcraft: error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
