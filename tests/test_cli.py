"""End-to-end tests of the ``foldcraft`` command-line interface."""

import io
import json

import pytest

import foldcraft as fc
from foldcraft.cli import main


def run(*argv):
    out = io.StringIO()
    return main([str(a) for a in argv], out), out.getvalue()


def test_info(mini_pdb):
    code, text = run("info", mini_pdb)
    assert code == 0 and "chain A: 8 residues" in text and "MKVLA" in text and "ligands ZN" in text
    data = json.loads(run("info", "--json", mini_pdb)[1])
    assert data["atoms"] == 36 and data["chains"][1]["sequence"] == "GG" and data["metadata"]["resolution"] == 1.8


def test_ss_contacts_and_clashes(tmp_path, helix, antiparallel_sheet, mini_pdb):
    fc.save(helix, tmp_path / "helix.pdb")
    fc.save(antiparallel_sheet, tmp_path / "sheet.cif")
    assert "-" + "H" * 14 + "-" in run("ss", tmp_path / "helix.pdb")[1]
    assert "C" + "H" * 14 + "C" in run("ss", "--three-state", tmp_path / "helix.pdb")[1]
    assert "res_i" in run("contacts", tmp_path / "sheet.cif", "--min-sep", "3")[1]
    assert "relative_contact_order" in run("contacts", tmp_path / "sheet.cif", "--summary")[1]
    code, text = run("clashes", mini_pdb)
    assert code == 0 and "clash_score:" in text


def test_compare(tmp_path):
    ref = fc.build_backbone("alpha", length=30)
    model = ref.copy()
    model.transform(fc.geometry.rotation_about_axis((0, 1, 0), 40.0), (3.0, 0.0, 0.0))
    fc.save(ref, tmp_path / "ref.pdb")
    fc.save(model, tmp_path / "model.cif")
    code, text = run("compare", tmp_path / "model.cif", tmp_path / "ref.pdb", "--json")
    data = json.loads(text)
    assert code == 0 and data["pairs"] == 30 and data["rmsd"] < 1e-2
    assert data["gdt_ts"] == pytest.approx(1.0) and data["tm_score"] == pytest.approx(1.0, abs=1e-3)
    assert "gdt_ha: 1.0" in run("compare", tmp_path / "model.cif", tmp_path / "ref.pdb", "--pairing", "numbering")[1]


def test_convert_round_trip(tmp_path, mini_pdb):
    assert run("convert", mini_pdb, tmp_path / "mini.cif")[0] == 0
    assert fc.load(tmp_path / "mini.cif").summary() == fc.load(mini_pdb).summary()
    assert run("convert", tmp_path / "mini.cif", tmp_path / "back.pdb")[0] == 0
    assert fc.load(tmp_path / "back.pdb").model["A"].sequence() == "MKVLA"


def test_errors_return_nonzero(tmp_path, capsys):
    bad = tmp_path / "bad.pdb"
    bad.write_text("ATOM      1  CA  ALA A   1       1.000   x.000   3.000\n")
    assert run("info", bad)[0] == 1 and "error" in capsys.readouterr().err
    assert run("info", tmp_path / "missing.pdb")[0] == 1
    assert run("--lenient", "info", bad)[0] == 1  # nothing parseable remains
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "foldcraft" in capsys.readouterr().out


def test_load_sniffs_unknown_extensions(tmp_path, mini_pdb, mini_cif):
    (tmp_path / "entry.txt").write_text(mini_cif.read_text())
    (tmp_path / "entry.dat").write_text(mini_pdb.read_text())
    assert fc.load(tmp_path / "entry.txt").id == "1FCY" and fc.load(tmp_path / "entry.dat").id == "1FCX"
    with pytest.raises(ValueError):
        fc.load(tmp_path / "entry.dat", fmt="xyz")
    with pytest.raises(ValueError):
        fc.save(fc.load(tmp_path / "entry.dat"), tmp_path / "x.out", fmt="xyz")
