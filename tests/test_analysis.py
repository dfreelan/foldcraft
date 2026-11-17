"""Contact maps, DSSP-style secondary structure, bond graphs and clash detection."""

import pytest

import foldcraft as fc
from foldcraft import geometry as geo
from foldcraft.contacts import representative_atom
from foldcraft.graph import MolecularGraph
from foldcraft.secondary import backbone_hbonds, hbond_energy
from foldcraft.structure import Atom, Chain, Model, Residue, Structure

from .conftest import atom_line


def shift_from(chain, index, delta):
    for res in chain.residues[index:]:
        for atom in res:
            atom.coord = geo.add(atom.coord, delta)


def test_distance_matrix_and_helix_contacts(helix):
    m = fc.distance_matrix([(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 0.0, 2.0)])
    assert m[0][1] == m[1][0] == 5.0 and all(m[i][i] == 0.0 for i in range(3))
    cmap = fc.ContactMap.from_structure(helix, cutoff=8.0, mode="CA")
    assert (0, 1) in cmap and (1, 0) in cmap and (0, 4) in cmap and (0, 15) not in cmap
    assert len(cmap.by_range("long")) == 0 and "x" not in cmap


def test_sheet_contacts_form_a_residue_graph(antiparallel_sheet):
    cmap = fc.ContactMap.from_structure(antiparallel_sheet, cutoff=6.0, mode="CA")
    assert any(i < 8 <= j for i, j in cmap) and cmap.labels[8] == "B:1"
    assert cmap.degrees()[0] == len(cmap.neighbors(0)) and any(j >= 8 for j in cmap.neighbors(0))


@pytest.mark.parametrize("mode", ["CA", "CB", "heavy"])
def test_contact_modes(helix, mode):
    cmap = fc.ContactMap.from_structure(helix, cutoff=7.0, mode=mode, min_separation=3)
    mat = cmap.to_matrix()
    assert len(cmap) > 0 and all(j - i >= 3 for i, j in cmap)
    assert all(mat[i][j] == mat[j][i] for i in range(cmap.n) for j in range(cmap.n))
    assert fc.ContactMap.from_structure(helix, 6.0, "CA").contacts <= fc.ContactMap.from_structure(helix, 6.0, "heavy").contacts


def test_representative_atom_fallbacks(helix):
    res = helix.model["A"][1]
    assert representative_atom(res, "CB") == res["CB"].coord
    res.remove_atom("CB")
    assert representative_atom(res, "CB") == res["CA"].coord
    with pytest.raises(ValueError):
        representative_atom(res, "XX")


def test_contact_statistics_and_rendering():
    ref = fc.ContactMap(10, [(0, 5), (1, 6), (2, 9)])
    stats = fc.ContactMap(10, [(5, 0), (1, 6), (3, 8), (4, 9)]).compare(ref)
    assert (stats.true_positives, stats.false_positives, stats.false_negatives) == (2, 2, 1)
    assert (stats.precision, stats.recall, stats.f1) == pytest.approx((0.5, 2 / 3, 4 / 7))
    empty = fc.ContactMap(3, []).compare(fc.ContactMap(3, []))
    assert empty.precision == empty.recall == empty.f1 == 0.0
    cmap = fc.ContactMap(4, [(0, 1), (0, 3)], distances={(0, 1): 3.8})
    assert cmap.relative_contact_order() == pytest.approx(0.5) and fc.ContactMap(4, []).relative_contact_order() == 0
    tsv = cmap.to_tsv().splitlines()
    assert tsv[0].startswith("i\tj") and tsv[1] == "1\t2\t1\t2\t3.800" and tsv[2].endswith("\t")
    assert len(cmap.filter(min_separation=2)) == len(cmap.filter(max_separation=1)) == 1
    for bad in (lambda: fc.ContactMap(3, [(0, 3)]), lambda: fc.ContactMap(3, [(1, 1)]),
                lambda: fc.ContactMap(3, [], labels=["a"]), lambda: ref.compare(fc.ContactMap(5, []))):
        with pytest.raises(ValueError):
            bad()


def test_alpha_and_310_helices(helix):
    ss = fc.assign_dssp(helix)
    assert ss.string == "-" + "H" * 14 + "-" and ss.three_state() == "C" + "H" * 14 + "C"
    assert ss.segments() == [("-", 0, 0), ("H", 1, 14), ("-", 15, 15)]
    assert ss.composition()["helix"] == pytest.approx(14 / 16)
    assert all(j - i in (3, 4) for i, j in ss.hbonds) and sum(j - i == 4 for i, j in ss.hbonds) >= 11
    s310 = fc.assign_dssp(fc.build_backbone("310", length=14)).string
    assert "G" in s310 and "H" not in s310


@pytest.mark.parametrize("orientation, kind", [("antiparallel", "A"), ("parallel", "P")])
def test_two_strand_sheets(orientation, kind):
    ss = fc.assign_dssp(fc.build_sheet(8, orientation))
    assert ss.string.count("E") >= 10 and {b[2] for b in ss.bridges} == {kind}
    assert all(ss.residues[i].parent.id != ss.residues[j].parent.id for i, j, _ in ss.bridges)


def test_isolated_strand_breaks_and_incomplete_residues(strand, helix):
    ss = fc.assign_dssp(strand)
    assert set(ss.string) <= {"-", "S", "T"} and not ss.hbonds
    shift_from(helix.model["A"], 8, (40.0, 0.0, 0.0))
    broken = fc.assign_dssp(helix)
    assert all(not (i < 8 <= j) for i, j in backbone_hbonds(broken.residues))
    helix.model["A"][5].remove_atom("O")
    assert len(fc.assign_dssp(helix)) == 15 and len(fc.assign_dssp([])) == 0


def test_hbond_energy():
    n, h, o, c = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.9, 0.0, 0.0), (4.13, 0.0, 0.0)
    assert hbond_energy(n, h, c, o) < -2.0
    assert hbond_energy(n, h, geo.add(c, (10.0, 0, 0)), geo.add(o, (10.0, 0, 0))) > -0.5
    assert hbond_energy(n, o, c, o) == -9.9  # coincident H and O hit the energy floor


def test_backbone_torsions(helix, strand):
    tors = fc.backbone_torsions(helix.model["A"])
    assert tors[0].phi is None and tors[-1].psi is None and tors[-1].omega is None
    for t in tors[1:-1]:
        assert (t.phi, t.psi, abs(t.omega)) == pytest.approx((-57.0, -47.0, 180.0))
    beta = fc.backbone_torsions(strand.model["A"])[3]
    assert (beta.phi, beta.psi) == pytest.approx((-120.0, 130.0))


def test_bond_graph(helix):
    g = fc.bond_graph(helix)
    assert g.n_bonds == 16 * 4 + 15  # N-CA, CA-C, C=O, CA-CB per residue plus peptide bonds
    assert [len(c) for c in g.connected_components()] == [80]
    ca = helix.model["A"][3]["CA"]
    assert {a.name for a in g.neighbors(ca)} == {"N", "C", "CB"}
    assert g.has_bond(helix.model["A"][3]["C"], helix.model["A"][4]["N"])
    assert len(g.within_bonds(g.index(ca), 1)) == 3 < len(g.within_bonds(g.index(ca), 2))
    assert all(i < j for i, j in g.bonds())
    with pytest.raises(ValueError):
        g.add_bond(0, 0)
    loose = MolecularGraph([Atom(str(i), "C", (float(i), 0.0, 0.0)) for i in range(5)])
    loose.add_bond(0, 1)
    loose.add_bond(3, 4)
    assert loose.connected_components() == [[0, 1], [3, 4], [2]]


def test_chain_breaks_and_disulfides(helix):
    chain = helix.model["A"]
    assert fc.find_chain_breaks(chain) == []
    shift_from(chain, 10, (0.0, 0.0, 10.0))
    chain[3].remove_atom("C")
    breaks = fc.find_chain_breaks(chain)
    assert [(b.before.seq, b.after.seq) for b in breaks] == [(3, 4), (10, 11)]
    assert breaks[0].distance is None and breaks[1].distance > 2.0
    assert len(fc.bond_graph(chain).connected_components()) == 4  # two fragments + orphaned O(3)
    s = Structure("ss")
    cys = s.add_model(Model()).add_chain(Chain("A"))
    for seq, x in ((1, 0.0), (2, 2.05)):
        res = cys.add_residue(Residue("CYS", seq))
        res.add_atom(Atom("SG", "S", (x, 0.0, 0.0)))
        res.add_atom(Atom("CB", "C", (x, 1.8 if seq == 1 else -1.8, 0.0)))
    assert fc.bond_graph(s).has_bond(cys[1]["SG"], cys[2]["SG"])


def test_ideal_structures_have_no_clashes(helix, antiparallel_sheet):
    assert len(fc.find_clashes(helix)) == 0 and fc.clash_score(antiparallel_sheet) == 0.0


def test_overlapping_chains_clash(helix):
    chain = helix.copy().model.chains.pop("A")
    chain.id = "B"
    helix.model.add_chain(chain)
    shift_from(chain, 0, (0.5, 0.0, 0.0))
    report = fc.find_clashes(helix)
    assert len(report) > 50 and report.clash_score > 100
    assert report.worst.overlap == max(c.overlap for c in report.clashes)
    assert report.worst.describe().startswith(("A:", "B:")) and " -- " in report.worst.describe()


def test_clash_tolerance_and_polar_allowance(nmr_pdb):
    lines = [atom_line(1, "C1", "LIG", "A", 1, (0.0, 0.0, 0.0), record="HETATM"),
             atom_line(2, "C1", "LIG", "B", 1, (2.9, 0.0, 0.0), record="HETATM"),
             atom_line(3, "O1", "HOH", "C", 1, (10.0, 0.0, 0.0), element="O", record="HETATM"),
             atom_line(4, "N1", "LIG", "D", 1, (12.6, 0.0, 0.0), element="N", record="HETATM")]
    s = fc.parse_pdb_string("\n".join(lines))
    report = fc.find_clashes(s)  # C...C at 2.9 < 3.4 - 0.4 clashes; O...N at 2.6 is an H-bond
    assert len(report) == 1 and report.clashes[0].overlap == pytest.approx(0.5)
    assert len(fc.find_clashes(s, hbond_allowance=0.0)) == 2
    assert len(fc.find_clashes(s, tolerance=0.5)) == len(fc.find_clashes(s, radii={"C": 1.0})) == 0
    with pytest.raises(ValueError):
        fc.find_clashes(s, tolerance=-1)
    nmr = fc.read_pdb(nmr_pdb)
    assert fc.find_clashes(nmr, include_hydrogens=True).n_atoms == fc.find_clashes(nmr).n_atoms + 1
    empty = Structure("e")
    empty.add_model(Model())
    assert fc.find_clashes(empty).n_atoms == 0
