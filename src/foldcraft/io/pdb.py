"""Reader and writer for the fixed-column PDB coordinate format.

The reader follows the wwPDB format v3.3 column layout and handles common
archive irregularities: hybrid-36 serials and residue numbers, blank element
columns (inferred from the atom-name justification convention), alternate
locations, insertion codes, multi-model entries, truncated lines and both
``2+`` / ``+2`` charge notations. In strict mode malformed records raise
:class:`~foldcraft.errors.PDBParseError`; otherwise they are skipped and
listed in ``structure.metadata["warnings"]``.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Iterable
from typing import TextIO

from ..constants import TWO_LETTER_ELEMENTS
from ..errors import PDBParseError, StructureError
from ..structure import Atom, Residue, Structure
from ._source import open_text, write_text
from .builder import AltlocPolicy, StructureBuilder

_DIGITS36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CRYST1_FIELDS = ((6, 15), (15, 24), (24, 33), (33, 40), (40, 47), (47, 54))


def hy36decode(width: int, text: str) -> int:
    """Decode a hybrid-36 field: decimal below ``10**width``, then upper- and lower-case base 36."""
    s = text.strip()
    if s and (s[0].isdigit() or s[0] == "-"):
        try:
            return int(s)
        except ValueError:
            pass
    elif len(s) == width and s.isalnum():
        if s[0].isupper() and not any(c.islower() for c in s):
            return int(s, 36) - 10 * 36 ** (width - 1) + 10**width
        if s[0].islower() and not any(c.isupper() for c in s):
            return int(s, 36) + 16 * 36 ** (width - 1) + 10**width
    raise ValueError(f"invalid hybrid-36 field {text!r}")


def hy36encode(width: int, value: int) -> str:
    """Encode *value* as a hybrid-36 field of the given *width*."""
    if -(10 ** (width - 1)) < value < 10**width:
        return f"{value:>{width}d}"
    span = 26 * 36 ** (width - 1)
    offset = value - 10**width
    for case in (str.upper, str.lower):
        if 0 <= offset < span:
            v, digits = offset + 10 * 36 ** (width - 1), ""
            while v:
                v, r = divmod(v, 36)
                digits = _DIGITS36[r] + digits
            return case(digits)
        offset -= span
    raise ValueError(f"value {value} out of hybrid-36 range for width {width}")


def infer_element(name_field: str, resname: str = "", hetero: bool = False) -> str:
    """Infer the element from the raw 4-character atom-name field (columns 13-16).

    One-letter elements are right-aligned into column 14 (``" CA "`` is a
    C-alpha carbon) while two-letter elements start in column 13 (``"CA  "``
    in a hetero group is calcium); a leading digit (``"1HB "``) is skipped.
    """
    field = name_field.ljust(4)[:4]
    if field[0] in " 0123456789":
        letters = [c for c in field[1:] if c.isalpha()]
        return letters[0].upper() if letters else ""
    pair = field[:2].upper()
    if pair in TWO_LETTER_ELEMENTS and (hetero or resname.upper() == pair):
        return pair
    return field[0].upper()


def parse_charge(field: str) -> int:
    """Parse a formal-charge field such as ``"2+"``, ``"1-"``, ``"+1"`` or ``"+"``."""
    s = field.strip()
    if not s:
        return 0
    sign = s[-1] if s[-1] in "+-" else s[0] if s[0] in "+-" else "+"
    digits = s.strip("+-") or "1"
    if not digits.isdigit() or len(s) - len(s.strip("+-")) > 1:
        raise ValueError(f"invalid charge field {field!r}")
    return int(digits) * (1 if sign == "+" else -1)


class PDBParser:
    """PDB reader configured with alternate-location, strictness and hydrogen policies."""

    def __init__(self, altloc: AltlocPolicy = "first", strict: bool = True, include_hydrogens: bool = True) -> None:
        self.altloc = altloc
        self.strict = strict
        self.include_hydrogens = include_hydrogens

    def parse_lines(self, lines: Iterable[str], structure_id: str = "", fallback_id: str = "") -> Structure:
        """Parse records; the identifier is *structure_id*, else the HEADER code, else *fallback_id*."""
        builder = StructureBuilder(structure_id, altloc=self.altloc, strict=self.strict)
        meta: dict[str, object] = {}
        titles: list[str] = []
        seqres: dict[str, list[str]] = {}
        model: int | None = None
        for lineno, raw in enumerate(lines, start=1):
            line = raw.rstrip("\r\n")
            record = line[:6].upper().strip()
            try:
                if record in ("ATOM", "HETATM"):
                    self._parse_atom(builder, line, lineno, model)
                elif record == "MODEL":
                    if model is not None:
                        raise PDBParseError("MODEL record without preceding ENDMDL", lineno, line)
                    model = int(line[6:].split()[0]) if line[6:].strip() else len(builder.structure) + 1
                    if any(m.serial == model for m in builder.structure):
                        raise PDBParseError(f"duplicate MODEL serial {model}", lineno, line)
                    builder.begin_model(model)
                elif record == "ENDMDL":
                    if model is None:
                        raise PDBParseError("ENDMDL record without MODEL", lineno, line)
                    model = None
                elif record == "HEADER":
                    meta.update(classification=line[10:50].strip(), deposition_date=line[50:59].strip(),
                                id_code=line[62:66].strip())
                elif record == "TITLE":
                    titles.append(line[10:80].strip())
                elif record == "EXPDTA":
                    meta["method"] = line[10:79].strip()
                elif record == "CRYST1":
                    meta["cell"] = tuple(float(line[a:b]) for a, b in _CRYST1_FIELDS)
                    meta["space_group"] = line[55:66].strip()
                elif record == "REMARK" and line[6:10].strip() == "2" and "RESOLUTION." in line:
                    meta["resolution"] = float(line.split("RESOLUTION.")[1].split()[0])
                elif record == "SEQRES":
                    seqres.setdefault(line[11:12].strip(), []).extend(line[19:80].split())
                elif record == "END":
                    break
            except (PDBParseError, ValueError, IndexError) as exc:
                err = exc if isinstance(exc, PDBParseError) else PDBParseError(f"malformed {record} record: {exc}", lineno, line)
                if self.strict:
                    raise err from None
                builder.warnings.append(str(err))
        if model is not None and self.strict:
            raise PDBParseError("file ended inside a MODEL block (missing ENDMDL)")
        meta.update(title=" ".join(titles), seqres=seqres)
        structure = builder.finish()
        if not structure.models:
            raise PDBParseError("no ATOM or HETATM records found")
        structure.id = structure.id or str(meta.get("id_code") or "") or fallback_id
        structure.metadata.update({k: v for k, v in meta.items() if v})
        if builder.warnings:
            structure.metadata["warnings"] = list(builder.warnings)
        return structure

    def _parse_atom(self, builder: StructureBuilder, line: str, lineno: int, model: int | None) -> None:
        def fail(message: str) -> PDBParseError:
            return PDBParseError(message, lineno, line)

        if len(line.rstrip()) < 54:
            raise fail("coordinate record shorter than 54 columns")
        p = line.ljust(80)
        hetero = p.startswith("HETATM")
        name, resname = p[12:16].strip(), p[17:20].strip()
        if not name:
            raise fail("blank atom name")
        try:
            serial = hy36decode(5, p[6:11]) if p[6:11].strip() else 0
            seq = hy36decode(4, p[22:26])
        except ValueError as exc:
            raise fail(str(exc)) from None
        try:
            coord = (float(p[30:38]), float(p[38:46]), float(p[46:54]))
        except ValueError:
            raise fail("non-numeric coordinates") from None
        try:
            occupancy = float(p[54:60]) if p[54:60].strip() else 1.0
            b_factor = float(p[60:66]) if p[60:66].strip() else 0.0
            charge = parse_charge(p[78:80])
        except ValueError as exc:
            raise fail(f"invalid occupancy, B-factor or charge ({exc})") from None
        if not 0.0 <= occupancy <= 1.0 + 1e-6:
            raise fail(f"occupancy {occupancy} outside [0, 1]")
        element = p[76:78].strip().upper() or infer_element(p[12:16], resname, hetero)
        if not self.include_hydrogens and element in ("H", "D"):
            return
        atom = Atom(name, element, coord, serial, occupancy, b_factor, p[16].strip(), charge)
        builder.add_atom(atom, p[21], resname, seq, p[26].strip(), hetero, model)


def parse_pdb_string(text: str, structure_id: str = "", **options: object) -> Structure:
    """Parse PDB-format *text*; keyword options are passed to :class:`PDBParser`."""
    return PDBParser(**options).parse_lines(text.splitlines(), structure_id)  # type: ignore[arg-type]


def read_pdb(source: str | os.PathLike[str] | TextIO, structure_id: str | None = None, **options: object) -> Structure:
    """Read a PDB file from a path (optionally gzip-compressed) or an open text handle."""
    with open_text(source) as (handle, stem):
        return PDBParser(**options).parse_lines(handle, structure_id or "", stem)  # type: ignore[arg-type]


def format_atom_name(name: str, element: str) -> str:
    """Pad an atom name into columns 13-16 following the PDB justification rules."""
    if len(name) > 4:
        raise StructureError(f"atom name {name!r} longer than 4 characters")
    return f"{name:<4}" if len(name) == 4 or len(element) == 2 else f" {name:<3}"


def _coord(value: float) -> str:
    text = f"{value:8.3f}"
    if len(text) > 8:
        raise StructureError(f"coordinate {value} does not fit the PDB format")
    return text


def _residue_fields(res: Residue, chain_id: str) -> str:
    if len(chain_id) > 1:
        raise StructureError(f"chain id {chain_id!r} cannot be written to PDB format (max 1 character)")
    return f"{res.name:>3} {chain_id or ' '}{hy36encode(4, res.seq)}{res.icode[:1] or ' '}"


def format_atom_line(atom: Atom, serial: int, res: Residue, chain_id: str) -> str:
    """Format one ATOM/HETATM record (at most 80 columns)."""
    charge = f"{abs(atom.charge)}{'+' if atom.charge > 0 else '-'}" if atom.charge else ""
    return (
        f"{'HETATM' if res.hetero else 'ATOM  '}{hy36encode(5, serial)} {format_atom_name(atom.name, atom.element)}"
        f"{atom.altloc[:1] or ' '}{_residue_fields(res, chain_id)}   "
        f"{_coord(atom.x)}{_coord(atom.y)}{_coord(atom.z)}{atom.occupancy:6.2f}{atom.b_factor:6.2f}"
        f"          {atom.element:>2}{charge}"
    ).rstrip()


def _header_lines(structure: Structure) -> list[str]:
    meta, out = structure.metadata, []
    id_code = str(meta.get("id_code", structure.id if len(structure.id) == 4 else ""))
    if id_code or meta.get("classification"):
        cls, date = str(meta.get("classification", ""))[:40], str(meta.get("deposition_date", ""))[:9]
        out.append(f"HEADER    {cls:<40}{date:<9}   {id_code[:4]}".rstrip())
    for n, chunk in enumerate(textwrap.wrap(str(meta.get("title", "")), 69), start=1):
        out.append(f"TITLE     {chunk}" if n == 1 else f"TITLE   {n:>2} {chunk}")
    cell = meta.get("cell")
    if isinstance(cell, tuple) and len(cell) == 6:
        a, b, c, al, be, ga = cell
        sg = str(meta.get("space_group", "P 1"))
        out.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {sg:<11}{1:>4}")
    return out


def to_pdb_string(structure: Structure) -> str:
    """Serialise *structure* as PDB text (header, coordinates with TER records, END)."""
    out = _header_lines(structure)
    multi = len(structure.models) > 1
    for model in structure:
        if multi:
            out.append(f"MODEL     {model.serial:>4}")
        serial = 1
        for chain in model:
            last_polymer: Residue | None = None
            for res in [*chain, None]:
                if last_polymer is not None and (res is None or res.hetero):
                    out.append(f"TER   {hy36encode(5, serial)}      {_residue_fields(last_polymer, chain.id)}")
                    serial, last_polymer = serial + 1, None
                if res is None:
                    break
                for atom in res:
                    out.append(format_atom_line(atom, serial, res, chain.id))
                    serial += 1
                if not res.hetero:
                    last_polymer = res
        if multi:
            out.append("ENDMDL")
    out.append("END")
    return "\n".join(out) + "\n"


def write_pdb(structure: Structure, target: str | os.PathLike[str] | TextIO) -> None:
    """Write *structure* in PDB format to a path or writable text handle."""
    write_text(to_pdb_string(structure), target)
