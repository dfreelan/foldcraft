"""Reader and writer for a practical subset of the PDBx/mmCIF format.

The tokenizer implements the CIF 1.1 lexical rules that matter for
macromolecular files: ``data_`` blocks, ``loop_`` constructs, quoted strings
in which a quote only terminates the value when followed by whitespace (so
``'O5''`` style atom names survive), semicolon text fields, ``#`` comments,
and the null markers ``?`` and ``.`` (mapped to ``None`` unless quoted).
Structures are built from ``atom_site``; ``struct``, ``exptl``, ``refine``,
``cell`` and ``symmetry`` populate :attr:`Structure.metadata`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple, TextIO

from ..errors import MMCIFParseError
from ..structure import Atom, Structure
from ._source import open_text, write_text
from .builder import AltlocPolicy, StructureBuilder

CIFValue = str | None
_RESERVED = ("data_", "save_", "global_", "stop_")
_CELL_ITEMS = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")


class Token(NamedTuple):
    value: str
    quoted: bool
    line: int


@dataclass
class CIFBlock:
    """A parsed ``data_`` block mapping category -> item -> column of values."""

    name: str
    categories: dict[str, dict[str, list[CIFValue]]] = field(default_factory=dict)

    def __contains__(self, category: object) -> bool:
        return category in self.categories

    def category(self, name: str) -> dict[str, list[CIFValue]]:
        try:
            return self.categories[name.lstrip("_")]
        except KeyError:
            raise MMCIFParseError(f"category {name!r} not present in block {self.name!r}") from None

    def rows(self, name: str) -> list[dict[str, CIFValue]]:
        """The category as a list of row dictionaries."""
        cols = self.category(name)
        n = len(next(iter(cols.values()), []))
        return [{k: v[i] for k, v in cols.items()} for i in range(n)]

    def value(self, category: str, item: str) -> CIFValue:
        """First value of ``_category.item``, or ``None``."""
        values = self.categories.get(category, {}).get(item)
        return values[0] if values else None


def tokenize(lines: Iterable[str]) -> Iterator[Token]:
    """Split CIF text into tokens following the STAR/CIF 1.1 lexical rules."""
    it = enumerate(lines, start=1)
    for lineno, raw in it:
        line = raw.rstrip("\r\n")
        if line.startswith(";"):
            parts = [line[1:]]
            for _, nxt in it:
                if nxt.startswith(";"):
                    break
                parts.append(nxt.rstrip("\r\n"))
            else:
                raise MMCIFParseError("unterminated semicolon text field", lineno, line)
            yield Token("\n".join(parts if parts[0] else parts[1:]), True, lineno)
            continue
        i, n = 0, len(line)
        while i < n:
            c = line[i]
            if c in " \t":
                i += 1
            elif c == "#":
                break
            elif c in "'\"":
                j = line.find(c, i + 1)
                while j != -1 and j + 1 < n and line[j + 1] not in " \t":
                    j = line.find(c, j + 1)
                if j == -1:
                    raise MMCIFParseError("unterminated quoted string", lineno, line)
                yield Token(line[i + 1 : j], True, lineno)
                i = j + 1
            else:
                j = i
                while j < n and line[j] not in " \t":
                    j += 1
                yield Token(line[i:j], False, lineno)
                i = j


def _is_keyword(tok: Token) -> bool:
    low = tok.value.lower()
    return not tok.quoted and (low == "loop_" or low.startswith(_RESERVED) or tok.value.startswith("_"))


def _split_tag(tok: Token) -> tuple[str, str]:
    cat, _, item = tok.value[1:].partition(".")
    if not cat or (tok.value.count(".") and not item):
        raise MMCIFParseError(f"malformed tag {tok.value!r}", tok.line)
    return cat, item


def parse_cif_blocks(lines: Iterable[str]) -> list[CIFBlock]:
    """Parse CIF text into :class:`CIFBlock` objects (save frames are flattened)."""
    tokens = list(tokenize(lines))
    blocks: list[CIFBlock] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.value.lower()
        i += 1
        if not tok.quoted and low.startswith("data_"):
            blocks.append(CIFBlock(tok.value[5:]))
            continue
        if not tok.quoted and low.startswith(_RESERVED[1:]):
            continue
        if not _is_keyword(tok):
            raise MMCIFParseError(f"unexpected value {tok.value!r} outside a loop", tok.line)
        if not blocks:
            raise MMCIFParseError("data item before any data_ block header", tok.line, tok.value)
        block = blocks[-1]
        if low == "loop_":
            tags = []
            while i < len(tokens) and not tokens[i].quoted and tokens[i].value.startswith("_"):
                tags.append(_split_tag(tokens[i]))
                i += 1
            if not tags:
                raise MMCIFParseError("loop_ without tags", tok.line)
            if len({c for c, _ in tags}) != 1:
                raise MMCIFParseError(f"loop_ mixes categories {sorted({c for c, _ in tags})}", tok.line)
            values: list[CIFValue] = []
            while i < len(tokens) and not _is_keyword(tokens[i]):
                values.append(_value(tokens[i]))
                i += 1
            if len(values) % len(tags):
                raise MMCIFParseError(f"loop_ for {tags[0][0]!r} has {len(values)} values, "
                                      f"not a multiple of {len(tags)} tags", tok.line)
            cols = block.categories.setdefault(tags[0][0], {})
            for k, (_, item) in enumerate(tags):
                cols.setdefault(item, []).extend(values[k :: len(tags)])
        else:
            if i >= len(tokens) or _is_keyword(tokens[i]):
                raise MMCIFParseError(f"tag {tok.value!r} has no value", tok.line)
            cat, item = _split_tag(tok)
            block.categories.setdefault(cat, {}).setdefault(item, []).append(_value(tokens[i]))
            i += 1
    return blocks


def _value(tok: Token) -> CIFValue:
    return None if not tok.quoted and tok.value in ("?", ".") else tok.value


def block_to_structure(block: CIFBlock, altloc: AltlocPolicy = "first", strict: bool = True,
                       use_author_fields: bool = True) -> Structure:
    """Build a :class:`Structure` from ``atom_site``, preferring ``auth_*`` identifiers by default."""
    if "atom_site" not in block:
        raise MMCIFParseError(f"block {block.name!r} has no atom_site category")
    builder = StructureBuilder(block.name, altloc=altloc, strict=strict)
    first, second = ("auth", "label") if use_author_fields else ("label", "auth")
    for idx, row in enumerate(block.rows("atom_site"), start=1):

        def pick(item: str) -> CIFValue:
            v = row.get(f"{first}_{item}")
            return v if v is not None else row.get(f"{second}_{item}")

        def num(item: str, default: float | None = None) -> float:
            v = row.get(item)
            if v is None and default is None:
                raise ValueError(f"missing {item}")
            return default if v is None else float(v)  # type: ignore[return-value]

        try:
            name, resname, chain_id = pick("atom_id"), pick("comp_id"), pick("asym_id")
            if name is None or resname is None or chain_id is None:
                raise ValueError("missing atom, residue or chain identifier")
            occupancy = num("occupancy", 1.0)
            if not 0.0 <= occupancy <= 1.0 + 1e-6:
                raise ValueError(f"occupancy {occupancy} outside [0, 1]")
            coord = (num("Cartn_x"), num("Cartn_y"), num("Cartn_z"))
            atom = Atom(name, (row.get("type_symbol") or name[0]).upper(), coord, int(row.get("id") or idx),
                        occupancy, num("B_iso_or_equiv", 0.0), row.get("label_alt_id") or "",
                        int(row.get("pdbx_formal_charge") or 0))
            seq = int(pick("seq_id") or 0)
            model = int(row.get("pdbx_PDB_model_num") or 1)
        except ValueError as exc:
            err = MMCIFParseError(f"atom_site row {idx}: {exc}")
            if strict:
                raise err from None
            builder.warnings.append(str(err))
            continue
        hetero = (row.get("group_PDB") or "ATOM").upper() == "HETATM"
        builder.add_atom(atom, chain_id, resname, seq, row.get("pdbx_PDB_ins_code") or "", hetero, model)

    structure = builder.finish()
    structure.models.sort(key=lambda m: m.serial)
    resolution = block.value("refine", "ls_d_res_high") or block.value("reflns", "d_resolution_high")
    cell = [block.value("cell", k) for k in _CELL_ITEMS]
    meta = {
        "title": " ".join(str(block.value("struct", "title") or "").split()),
        "method": block.value("exptl", "method"),
        "resolution": float(resolution) if resolution else None,
        "cell": tuple(float(v) for v in cell) if None not in cell else None,  # type: ignore[arg-type]
        "space_group": block.value("symmetry", "space_group_name_H-M"),
        "warnings": list(builder.warnings),
    }
    structure.metadata.update({k: v for k, v in meta.items() if v})
    return structure


def parse_mmcif_string(text: str, **options: object) -> Structure:
    """Parse mmCIF *text* and return the structure of its first data block."""
    return _first_structure(parse_cif_blocks(text.splitlines()), options)


def read_mmcif(source: str | os.PathLike[str] | TextIO, **options: object) -> Structure:
    """Read an mmCIF file from a path (optionally gzip-compressed) or text handle."""
    with open_text(source) as (handle, _):
        return _first_structure(parse_cif_blocks(handle), options)


def _first_structure(blocks: list[CIFBlock], options: dict[str, object]) -> Structure:
    if not blocks:
        raise MMCIFParseError("no data_ block found")
    return block_to_structure(blocks[0], **options)  # type: ignore[arg-type]


def quote_value(value: object) -> str:
    """Render a value as a CIF token, quoting only when required."""
    if value is None:
        return "?"
    text = str(value)
    if text == "":
        return "."
    plain = not any(c in text for c in " \t\n") and text[0] not in "_#$'\";[]" and text not in ("?", ".")
    if plain and not text.lower().startswith(_RESERVED + ("loop_",)):
        return text
    for q in "'\"":
        if "\n" not in text and f"{q} " not in text and not text.endswith(q):
            return f"{q}{text}{q}"
    return f"\n;{text}\n;\n"


_ATOM_SITE_ITEMS = (
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
    "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv",
    "pdbx_formal_charge", "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num",
)


def structure_to_mmcif(structure: Structure) -> str:
    """Serialise *structure* as a single mmCIF data block."""
    name = (structure.id or "foldcraft").replace(" ", "_")
    out, meta = [f"data_{name}", "#"], structure.metadata
    if meta.get("title"):
        out += [f"_struct.entry_id {quote_value(name)}", f"_struct.title {quote_value(meta['title'])}"]
    cell = meta.get("cell")
    if isinstance(cell, tuple) and len(cell) == 6:
        out += [f"_cell.{k} {v:.3f}" for k, v in zip(_CELL_ITEMS, cell)]
        out.append(f"_symmetry.space_group_name_H-M {quote_value(meta.get('space_group', 'P 1'))}")
    out += ["#", "loop_"] + [f"_atom_site.{item}" for item in _ATOM_SITE_ITEMS]
    serial = 1
    for model in structure:
        for chain in model:
            cid = quote_value(chain.id)
            for res in chain:
                for a in res:
                    out.append(" ".join((
                        "HETATM" if res.hetero else "ATOM", str(serial), a.element, quote_value(a.name),
                        quote_value(a.altloc), quote_value(res.name), cid, str(res.seq), quote_value(res.icode or None),
                        f"{a.x:.3f}", f"{a.y:.3f}", f"{a.z:.3f}", f"{a.occupancy:.2f}", f"{a.b_factor:.2f}",
                        str(a.charge), str(res.seq), cid, str(model.serial),
                    )))
                    serial += 1
    return "\n".join(out + ["#"]) + "\n"


def write_mmcif(structure: Structure, target: str | os.PathLike[str] | TextIO) -> None:
    """Write *structure* as mmCIF to a path or writable text handle."""
    write_text(structure_to_mmcif(structure), target)
