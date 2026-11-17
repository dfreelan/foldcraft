"""PDB reader / writer: columns, edge cases, validation and round trips."""

import gzip
import io

import pytest

import foldcraft as fc
from foldcraft.io import hy36decode, hy36encode, infer_element, parse_pdb_string, read_pdb, to_pdb_string
from foldcraft.io.pdb import format_atom_name, parse_charge

from .conftest import atom_line


def test_header_metadata_and_hierarchy(mini_pdb):
    s = read_pdb(mini_pdb)
    meta = s.metadata
    assert s.id == "1FCX" and meta["method"] == "X-RAY DIFFRACTION" and meta["resolution"] == 1.80
    assert meta["title"] == "SYNTHETIC HELICAL PENTAPEPTIDE WITH A BOUND ZINC ION AND A SHORT SECOND CHAIN"
    assert meta["cell"] == (40.0, 50.0, 60.0, 90.0, 90.0, 90.0) and meta["space_group"] == "P 21 21 21"
    assert meta["seqres"] == {"A": ["MET", "LYS", "VAL", "LEU", "ALA"], "B": ["GLY", "GLY"]}
    assert list(s.model.chains) == ["A", "B"]
    assert [str(r.id) for r in s.model["A"]] == ["1", "2", "3", "3A", "4", "101", "201", "202"]
    assert s.model["A"].sequence() == "MKVLA" and s.model["B"].sequence() == "GG"
    assert read_pdb(mini_pdb, structure_id="custom").id == "custom"


def test_elements_charges_hetero_groups_and_truncation(mini_pdb):
    atom = next(parse_pdb_string("ATOM      1  CA  GLY A   1      11.104   6.134  -6.504\r\nEND\r\n").atoms())
    assert (atom.occupancy, atom.b_factor, atom.element) == (1.0, 0.0, "C")
    chain = read_pdb(mini_pdb).model["A"]
    assert chain[2]["CB"].element == "C"  # blank element column, inferred from " CB "
    assert chain[101]["ZN"].element == "ZN" and chain[101]["ZN"].charge == 2
    assert chain[101].hetero and chain[201].is_water and chain[202]["O"].occupancy == 0.5


@pytest.mark.parametrize("policy, x, alt", [("first", -1.030, "A"), ("occupancy", -1.030, "A"), ("B", -1.130, "B")])
def test_altloc_policies(mini_pdb, policy, x, alt):
    cb = read_pdb(mini_pdb, altloc=policy).model["A"][3]["CB"]
    assert (cb.x, cb.altloc) == (pytest.approx(x), alt)


def test_altloc_occupancy_prefers_higher():
    lines = "\n".join([atom_line(1, "CA", "SER", "A", 1, (0, 0, 0), altloc="A", occupancy=0.3),
                       atom_line(2, "CA", "SER", "A", 1, (1, 0, 0), altloc="B", occupancy=0.7)])
    assert parse_pdb_string(lines, altloc="occupancy").model["A"][1]["CA"].altloc == "B"
    with pytest.raises(ValueError):
        parse_pdb_string(lines, altloc="AB")


def test_multi_model_and_hydrogen_filter(nmr_pdb):
    s = read_pdb(nmr_pdb)
    assert [m.serial for m in s] == [1, 2] and s[1]["A"][1]["N"].x == pytest.approx(0.1)
    assert sum(1 for _ in s.all_atoms()) == 18
    assert all(not a.is_hydrogen for a in read_pdb(nmr_pdb, include_hydrogens=False).all_atoms())


BASE = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C"


@pytest.mark.parametrize("text, message", [
    (BASE[:46], "shorter than 54"),
    (BASE[:38] + "   x.000" + BASE[46:], "non-numeric coordinates"),
    (BASE[:54] + "  1.50" + BASE[60:], "occupancy"),
    (BASE[:12] + "    " + BASE[16:], "blank atom name"),
    (BASE + "+X", "charge"),
    (BASE[:22] + " 1?2" + BASE[26:], "hybrid-36"),
    ("HEADER    NOTHING HERE", "no ATOM or HETATM"),
    ("\n".join(["MODEL        1", BASE, "MODEL        2", BASE]), "without preceding ENDMDL"),
    ("\n".join([BASE, "ENDMDL"]), "ENDMDL record without MODEL"),
    ("\n".join(["MODEL        1", BASE]), "missing ENDMDL"),
    ("\n".join(["MODEL        1", BASE, "ENDMDL", "MODEL        1", BASE, "ENDMDL"]), "duplicate MODEL"),
    (BASE + "\nCRYST1   40.000   abc", "malformed CRYST1"),
])
def test_strict_errors(text, message):
    with pytest.raises(fc.PDBParseError, match=message):
        parse_pdb_string(text)


def test_error_line_numbers_and_lenient_mode():
    with pytest.raises(fc.PDBParseError) as info:
        parse_pdb_string("\n".join([BASE, BASE[:38] + "   x.000" + BASE[46:]]))
    assert info.value.line_number == 2 and "line 2" in str(info.value)
    dup = BASE.replace("   1       1.000", "   1       9.000")
    s = parse_pdb_string("\n".join([BASE, dup, BASE[:30]]), strict=False)
    assert len(list(s.atoms())) == 1 and len(s.metadata["warnings"]) == 2
    with pytest.raises(fc.StructureError, match="duplicate atom"):
        parse_pdb_string("\n".join([BASE, dup]))


def test_microheterogeneity_keeps_first_residue_type():
    lines = [atom_line(1, "CA", "SER", "A", 7, (0, 0, 0), altloc="A", occupancy=0.5),
             atom_line(2, "CA", "THR", "A", 7, (0.2, 0, 0), altloc="B", occupancy=0.5),
             atom_line(3, "CB", "THR", "A", 7, (1.2, 0, 0), altloc="B", occupancy=0.5)]
    s = parse_pdb_string("\n".join(lines), strict=False)
    assert s.model["A"][7].name == "SER" and list(s.model["A"][7].atoms) == ["CA"]
    assert any("modelled as both" in w for w in s.metadata["warnings"])


def test_round_trip_is_stable(mini_pdb, nmr_pdb):
    s1 = read_pdb(mini_pdb)
    text = to_pdb_string(s1)
    s2 = parse_pdb_string(text)
    assert to_pdb_string(s2) == text and s2.id == "1FCX"
    for key in ("cell", "title", "space_group"):
        assert s2.metadata[key] == s1.metadata[key]
    for a, b in zip(s1.all_atoms(), s2.all_atoms(), strict=True):
        assert (a.name, a.element, a.altloc, a.charge, a.parent.id, a.parent.hetero) == \
               (b.name, b.element, b.altloc, b.charge, b.parent.id, b.parent.hetero)
        assert a.coord == pytest.approx(b.coord, abs=1e-3)
    nmr = to_pdb_string(read_pdb(nmr_pdb))
    assert nmr.count("MODEL") == nmr.count("ENDMDL") == 2 and len(parse_pdb_string(nmr)) == 2


def test_writer_layout_and_limits(mini_pdb, helix):
    text = to_pdb_string(read_pdb(mini_pdb))
    lines = {line[12:16]: line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))}
    assert " CA " in lines and lines["ZN  "][76:80] == "ZN2+"
    assert all(len(line) <= 80 for line in text.splitlines()) and text.count("TER") == 2
    next(helix.atoms()).coord = (123456.0, 0.0, 0.0)
    with pytest.raises(fc.StructureError, match="does not fit"):
        to_pdb_string(helix)
    helix.model.chains["A"].id = "AB"
    with pytest.raises(fc.StructureError, match="chain id"):
        to_pdb_string(helix)


def test_gzip_and_handle_input(tmp_path, mini_pdb):
    gz = tmp_path / "mini.pdb.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write(mini_pdb.read_text())
    assert read_pdb(gz).summary()["atoms"] == 36
    assert read_pdb(io.StringIO(mini_pdb.read_text())).summary() == read_pdb(gz).summary()


@pytest.mark.parametrize("value", [0, 99999, 100000, 1_000_000, 2_436_111, 2_436_112, 43_770_015])
def test_hybrid36_round_trip(value):
    assert hy36decode(5, hy36encode(5, value)) == value


def test_hybrid36_known_values_and_residue_numbers():
    assert hy36encode(5, 100000) == "A0000" and hy36encode(4, 10000) == "A000"
    assert hy36decode(4, "a000") == 10000 + 26 * 36**3 and hy36decode(4, "  -5") == -5
    for bad in ("", "A0a0", "A00", "A0!00"):
        with pytest.raises(ValueError):
            hy36decode(5, bad)
    with pytest.raises(ValueError):
        hy36encode(4, 10**8)
    s = parse_pdb_string(BASE[:22] + "A000" + BASE[26:])
    assert next(s.residues()).seq == 10000 and "A000" in to_pdb_string(s)


@pytest.mark.parametrize("field, resname, hetero, expected", [
    (" CA ", "ALA", False, "C"), ("CA  ", "CA", True, "CA"), ("1HB ", "ALA", False, "H"), ("HD21", "ASN", False, "H"),
    ("FE  ", "HEM", True, "FE"), (" OG1", "THR", False, "O"), ("CL1 ", "LIG", True, "CL"), ("CD1 ", "LEU", False, "C"),
])
def test_infer_element(field, resname, hetero, expected):
    assert infer_element(field, resname, hetero) == expected


@pytest.mark.parametrize("field, expected", [("  ", 0), ("2+", 2), ("1-", -1), ("+1", 1), ("-2", -2), ("+", 1), ("3", 3)])
def test_parse_charge(field, expected):
    assert parse_charge(field) == expected


def test_format_atom_name_justification():
    assert (format_atom_name("CA", "C"), format_atom_name("CA", "CA"), format_atom_name("HD21", "H")) == \
           (" CA ", "CA  ", "HD21")
    with pytest.raises(fc.StructureError):
        format_atom_name("TOOLONG", "C")
