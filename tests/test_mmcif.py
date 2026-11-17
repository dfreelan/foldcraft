"""mmCIF tokenizer, block parser, structure construction and writer."""

import pytest

import foldcraft as fc
from foldcraft.io import parse_cif_blocks, parse_mmcif_string, read_mmcif, read_pdb, structure_to_mmcif, tokenize
from foldcraft.io.mmcif import quote_value


def values(text):
    return [(t.value, t.quoted) for t in tokenize(text.splitlines())]


def test_tokenizer_rules():
    assert values("_a.b 'it''s' \"x y\" plain # comment\n'O5'' end") == [
        ("_a.b", False), ("it''s", True), ("x y", True), ("plain", False), ("O5'", True), ("end", False)]
    toks = values("_a.b\n;line one\nline two\n;\n_c.d 1")
    assert toks[1] == ("line one\nline two", True) and toks[2:] == [("_c.d", False), ("1", False)]


@pytest.mark.parametrize("text, message", [
    ("data_x\n_a.b 'oops", "unterminated quoted"),
    ("data_x\n_a.b\n;never closed\n", "unterminated semicolon"),
    ("data_x\nloop_\n_a.x\n_a.y\n1 2 3", "not a multiple"),
    ("data_x\nloop_\n_a.x\n_b.y\n1 2", "mixes categories"),
    ("data_x\nloop_\n1 2", "without tags"),
    ("data_x\n_a.b\n_a.c 1", "has no value"),
    ("_a.b 1", "before any data_"),
    ("data_x\nstray", "outside a loop"),
    ("data_x\n_.b 1", "malformed tag"),
])
def test_parser_errors(text, message):
    with pytest.raises(fc.MMCIFParseError, match=message):
        parse_cif_blocks(text.splitlines())


def test_nulls_and_multiple_blocks():
    blocks = parse_cif_blocks("data_one\n_a.x ?\n_a.y .\n_a.z '?'\ndata_two\nloop_\n_b.k\n1\n2\n".splitlines())
    assert [b.name for b in blocks] == ["one", "two"]
    assert (blocks[0].value("a", "x"), blocks[0].value("a", "y"), blocks[0].value("a", "z")) == (None, None, "?")
    assert blocks[1].rows("b") == [{"k": "1"}, {"k": "2"}]
    with pytest.raises(fc.MMCIFParseError):
        blocks[1].category("missing")


def test_read_with_author_fields(mini_cif):
    s = read_mmcif(mini_cif)
    meta = s.metadata
    assert s.id == "1FCY" and meta["title"] == "Synthetic tripeptide bound to an adenosine fragment"
    assert meta["method"] == "SOLUTION NMR" and meta["resolution"] == 2.1
    assert meta["cell"][5] == 120.0 and meta["space_group"] == "P 31 2 1"
    chain = s.model["X"]
    assert [r.seq for r in chain] == [10, 11, 12, 501] and chain.sequence() == "MKV"
    assert chain[501].hetero and set(chain[501].atoms) == {"C4'", "C5'", "N1"} and chain[501]["N1"].charge == -1


def test_label_fields_and_altlocs(mini_cif):
    s = read_mmcif(mini_cif, use_author_fields=False)
    assert list(s.model.chains) == ["A", "B"] and [r.seq for r in s.model["A"]] == [1, 2, 3]
    assert read_mmcif(mini_cif).model["X"][12]["CB"].altloc == "A"
    assert read_mmcif(mini_cif, altloc="occupancy").model["X"][12]["CB"].altloc == "B"


def test_bad_rows_and_missing_content():
    head = "data_t\nloop_\n" + "".join(f"_atom_site.{k}\n" for k in (
        "group_PDB", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"))
    text = head + "ATOM CA ALA A 1 1.0 2.0 3.0\nATOM CB ALA A 1 1.0 abc 3.0\n"
    with pytest.raises(fc.MMCIFParseError, match="row 2"):
        parse_mmcif_string(text)
    s = parse_mmcif_string(text, strict=False)
    assert len(list(s.atoms())) == 1 and len(s.metadata["warnings"]) == 1
    with pytest.raises(fc.MMCIFParseError, match="no atom_site"):
        parse_mmcif_string("data_t\n_a.b 1\n")
    with pytest.raises(fc.MMCIFParseError, match="no data_ block"):
        parse_mmcif_string("# empty\n")


def test_round_trips(mini_cif, mini_pdb, nmr_pdb):
    s1 = read_mmcif(mini_cif)
    s2 = parse_mmcif_string(structure_to_mmcif(s1))
    assert [(a.name, a.coord) for a in s1.all_atoms()] == [(a.name, a.coord) for a in s2.all_atoms()]
    assert (s2.metadata["title"], s2.metadata["space_group"]) == (s1.metadata["title"], "P 31 2 1")
    p1 = read_pdb(mini_pdb)
    p2 = parse_mmcif_string(structure_to_mmcif(p1))
    assert p2.summary() == p1.summary()
    for a, b in zip(p1.all_atoms(), p2.all_atoms(), strict=True):
        assert (a.name, a.element, a.charge, a.occupancy, a.parent.name, a.parent.id) == \
               (b.name, b.element, b.charge, b.occupancy, b.parent.name, b.parent.id)
    assert [m.serial for m in parse_mmcif_string(structure_to_mmcif(read_pdb(nmr_pdb)))] == [1, 2]


@pytest.mark.parametrize("value, expected", [
    (None, "?"), ("", "."), ("CA", "CA"), ("C4'", "C4'"), ("a b", "'a b'"), ("it's here", "'it's here'"),
    ("x' y", "\"x' y\""), ("?", "'?'"), ("_x", "'_x'"), ("data_x", "'data_x'"),
])
def test_quote_value(value, expected):
    assert quote_value(value) == expected


@pytest.mark.parametrize("text", ["a b", "it's here", "x' y\" z", "multi\nline"])
def test_quoted_values_survive_tokenizer(text):
    assert parse_cif_blocks(f"data_q\n_t.v {quote_value(text)}\n".splitlines())[0].value("t", "v") == text
