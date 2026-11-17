"""Geometry, spatial hashing, the structure hierarchy and residue pairing."""

import math
import random

import pytest

import foldcraft as fc
from foldcraft import geometry as geo
from foldcraft.structure import Atom, Chain, Model, Residue, ResidueId, Structure


def make_residue(name="ALA", seq=1, icode=""):
    res = Residue(name, seq, icode)
    for i, atom_name in enumerate(("N", "CA", "C", "O")):
        res.add_atom(Atom(atom_name, atom_name[0], (float(i), 0.0, 0.0)))
    return res


def test_vector_operations():
    a, b = (1.0, 2.0, 3.0), (4.0, -1.0, 0.5)
    assert geo.add(a, b) == (5.0, 1.0, 3.5) and geo.sub(a, b) == (-3.0, 3.0, 2.5)
    assert geo.dot(geo.cross(a, b), a) == pytest.approx(0.0)
    assert geo.norm(geo.normalize(b)) == pytest.approx(1.0)
    assert geo.distance((0, 0, 0), (3, 4, 0)) == geo.distance_sq((0, 0, 0), (3, 4, 0)) ** 0.5 == 5.0
    assert geo.angle((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90.0)
    for bad in (lambda: geo.normalize((0, 0, 0)), lambda: geo.angle((0, 0, 0), (0, 0, 0), (1, 0, 0)),
                lambda: geo.dihedral((1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 1)), lambda: geo.centroid([]),
                lambda: geo.centroid([(0, 0, 0)], [1.0, 2.0]), lambda: geo.centroid([(0, 0, 0)], [0.0])):
        with pytest.raises(ValueError):
            bad()
    a, b, c = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    assert geo.dihedral(a, b, c, (0.0, 1.0, 1.0)) == pytest.approx(90.0)
    assert geo.dihedral(a, b, c, (0.0, -1.0, 1.0)) == pytest.approx(-90.0)
    assert geo.centroid([(0, 0, 0), (2, 0, 0)], [3.0, 1.0]) == pytest.approx((0.5, 0.0, 0.0))


@pytest.mark.parametrize("target", [-179.0, -60.0, 0.0, 45.0, 150.0, 180.0])
def test_place_atom_reproduces_internal_coordinates(target):
    a, b, c = (0.3, -1.2, 0.5), (1.1, 0.2, -0.4), (2.4, 0.1, 0.3)
    d = geo.place_atom(a, b, c, 1.53, 109.5, target)
    assert geo.distance(c, d) == pytest.approx(1.53) and geo.angle(b, c, d) == pytest.approx(109.5)
    assert math.cos(math.radians(geo.dihedral(a, b, c, d) - target)) == pytest.approx(1.0)


def test_rotation_and_jacobi():
    r = geo.rotation_about_axis((1.0, 1.0, 0.0), 37.0)
    assert geo.determinant(r) == pytest.approx(1.0)
    assert geo.mat_vec(geo.rotation_about_axis((0, 0, 1), 90.0), (1, 0, 0)) == pytest.approx((0, 1, 0))
    rng = random.Random(7)
    m = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i, 4):
            m[i][j] = m[j][i] = rng.uniform(-3, 3)
    values, vectors = geo.jacobi_eigh(m)
    assert values == sorted(values, reverse=True)
    for k, lam in enumerate(values):
        v = [vectors[i][k] for i in range(4)]
        assert [sum(m[i][j] * v[j] for j in range(4)) for i in range(4)] == pytest.approx([lam * x for x in v], abs=1e-9)
    with pytest.raises(ValueError):
        geo.jacobi_eigh([[1.0, 2.0], [3.0]])


@pytest.mark.parametrize("cutoff", [1.5, 4.0])
def test_grid_pairs_match_brute_force(cutoff):
    rng = random.Random(42)
    pts = [(rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10)) for _ in range(150)]
    brute = [(i, j) for i in range(150) for j in range(i + 1, 150) if geo.distance(pts[i], pts[j]) <= cutoff]
    assert [(i, j) for i, j, _ in fc.neighbor_pairs(pts, cutoff)] == brute


def test_grid_query_and_validation():
    grid = fc.CellGrid([(float(i), 0.0, 0.0) for i in range(20)], 1.0, items=[f"p{i}" for i in range(20)])
    assert sorted(item for item, _ in grid.query((0.0, 0.0, 0.0), 5.5)) == sorted(f"p{i}" for i in range(6))
    assert fc.neighbor_pairs([(-0.1, 0, 0), (0.1, 0, 0), (-5, -5, -5)], 0.5)[0][:2] == (0, 1)
    assert fc.neighbor_pairs([], 3.0) == []
    for bad in (lambda: list(grid.pairs(2.0)), lambda: fc.CellGrid([(0, 0, 0)], 0.0),
                lambda: fc.CellGrid([(0, 0, 0)], 1.0, items=[1, 2])):
        with pytest.raises(ValueError):
            bad()


def test_residue_and_chain_access():
    res = make_residue()
    assert res["CA"].coord == (1.0, 0.0, 0.0) and "CB" not in res and len(res) == 4
    assert res.has_backbone() and res.is_amino_acid and res.get("CB") is None
    with pytest.raises(fc.StructureError):
        res.add_atom(Atom("CA", "C", (0.0, 0.0, 0.0)))
    assert res.remove_atom("O").parent is None
    with pytest.raises(fc.StructureError):
        res["O"]
    chain = Chain("A")
    chain.add_residue(make_residue("GLY", 10))
    chain.add_residue(make_residue("SER", 10, "A"))
    assert chain[10].name == "GLY" and chain[(10, "A")].name == chain[ResidueId(10, "A")].name == "SER"
    assert (10, "A") in chain and 11 not in chain and "x" not in chain
    assert chain.sequence() == "GS" and str(chain[(10, "A")].id) == "10A"
    with pytest.raises(fc.StructureError):
        chain.add_residue(make_residue("ALA", 10))
    with pytest.raises(TypeError):
        chain["ten"]
    chain.remove_residue((10, "A"))
    assert len(chain) == 1


def test_hierarchy_links_and_errors():
    s = Structure("x")
    m = s.add_model(Model(1))
    c = m.add_chain(Chain("B"))
    atom = c.add_residue(make_residue("LYS", 5))["CA"]
    assert atom.label() == "B:LYS5:CA" and Atom("X", "C", (0, 0, 0)).label() == "X"
    assert atom.parent.parent.parent.parent is s
    for bad in (lambda: s.add_model(Model(1)),
                lambda: m.add_chain(Chain("B")), lambda: m["Z"], lambda: Structure().model):
        with pytest.raises(fc.StructureError):
            bad()
    water = Residue("HOH", 1, hetero=True)
    assert water.is_water and not water.is_amino_acid and Residue("MSE", 1).one_letter == "M"


def test_summary_transform_and_copy(helix):
    assert helix.summary() == {"models": 1, "chains": 1, "residues": 16, "atoms": 80}
    assert Structure().summary()["atoms"] == 0 and len(fc.ca_atoms(helix)) == 16
    clone = helix.copy()
    clone.transform(translation=(1.0, 2.0, 3.0))
    a0, a1 = next(helix.atoms()), next(clone.atoms())
    assert a1.coord == pytest.approx((a0.x + 1.0, a0.y + 2.0, a0.z + 3.0))
    assert a1.parent.parent.parent.parent is clone


def test_sequence_pairing_strategies(helix):
    path, score = fc.needleman_wunsch("ACDEF", "ACDEF")
    assert path == [(i, i) for i in range(5)] and score == 10.0
    path, _ = fc.needleman_wunsch("ACDEFGH", "ACDGH")
    assert sum(1 for _, j in path if j is None) == 2 and (6, 4) in path
    assert fc.needleman_wunsch("", "AC") == ([(None, 0), (None, 1)], -4.0)
    other = helix.copy()
    other.model["A"].remove_residue(5)
    assert len(fc.pair_residues(other, helix, "numbering")) == 15
    ref = fc.build_backbone("alpha", sequence="MKTAYIAKQRQISFVK")
    model = fc.build_backbone("alpha", sequence="MKTAYIAKQISFVK", start=101, chain_id="Z")
    pairs = fc.pair_residues(model, ref, "sequence")
    assert len(pairs) == 14 and all(a.name == b.name for a, b in pairs)
    assert fc.pair_residues(model, ref, "numbering") == []
    assert fc.pair_residues(model, ref, "order")[-1][1].seq == 14
    with pytest.raises(ValueError):
        fc.pair_residues(model, ref, "magic")


def test_paired_coordinates_skips_missing_atoms(helix):
    pairs = fc.pair_residues(helix, helix)
    helix.model["A"][2].remove_atom("CB")
    mob, ref = fc.paired_coordinates(pairs, ("CA", "CB"))
    assert len(mob) == len(ref) == 31
    with pytest.raises(fc.ComparisonError):
        fc.paired_coordinates(pairs, ("XX",))
