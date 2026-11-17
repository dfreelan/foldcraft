"""Kabsch superposition, RMSD and model-quality scores."""

import random

import pytest

import foldcraft as fc
from foldcraft import geometry as geo
from foldcraft.scores import maximal_subset_search, tm_d0
from foldcraft.superpose import quaternion_to_matrix


def random_points(n, seed=0):
    rng = random.Random(seed)
    return [(rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-15, 15)) for _ in range(n)]


def perturbed(points, sigma, seed=0):
    rng = random.Random(seed)
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), z + rng.gauss(0, sigma)) for x, y, z in points]


@pytest.fixture
def ref():
    return [a.coord for a in fc.ca_atoms(fc.build_backbone("alpha", length=40))]


@pytest.mark.parametrize("seed", range(4))
def test_kabsch_recovers_rigid_transform(seed):
    rng = random.Random(seed)
    pts = random_points(25, seed)
    rot = geo.rotation_about_axis((rng.uniform(-1, 1), rng.uniform(-1, 1), 0.5), rng.uniform(-180, 180))
    shift = (rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-30, 30))
    target = [geo.add(geo.mat_vec(rot, p), shift) for p in pts]
    sup = fc.kabsch(pts, target)
    assert sup.rmsd == pytest.approx(0.0, abs=1e-8) and sup.n_points == 25
    assert [x for row in sup.rotation for x in row] == pytest.approx([x for row in rot for x in row], abs=1e-8)
    assert sup.translation == pytest.approx(shift, abs=1e-7)
    assert sup.apply_all(pts)[3] == pytest.approx(target[3], abs=1e-8)


def test_kabsch_never_reflects_and_respects_weights():
    pts = random_points(12, 3)
    sup = fc.kabsch(pts, [(-x, y, z) for x, y, z in pts])
    assert geo.determinant(sup.rotation) == pytest.approx(1.0) and sup.rmsd > 1.0
    target = list(pts)
    target[0] = geo.add(target[0], (50.0, 0.0, 0.0))
    assert fc.kabsch(pts, target, [0.0] + [1.0] * 11).rmsd == pytest.approx(0.0, abs=1e-7)
    assert fc.kabsch(pts, target).rmsd > 1.0
    half = quaternion_to_matrix((0.0, 0.0, 0.0, 2.0))
    assert geo.mat_vec(half, (1.0, 0.0, 0.0)) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)


def test_rmsd_values():
    a, b = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    assert fc.rmsd(a, b) == fc.rmsd(a, b, [1.0, 3.0]) == pytest.approx(1.0)
    assert fc.superposed_rmsd(a, b) == pytest.approx(0.0, abs=1e-12)
    pts = random_points(40, 11)
    noisy = perturbed([geo.mat_vec(geo.rotation_about_axis((0.2, 1, -0.4), 63.0), p) for p in pts], 0.5, 5)
    assert 0.3 < fc.superposed_rmsd(pts, noisy) < 1.2 < fc.rmsd(pts, noisy)


@pytest.mark.parametrize("a, b, w", [([], [], None), ([(0, 0, 0)], [], None),
                                      ([(0, 0, 0)], [(1, 1, 1)], [1.0, 2.0]), ([(0, 0, 0)], [(1, 1, 1)], [-1.0])])
def test_superposition_input_validation(a, b, w):
    with pytest.raises(fc.ComparisonError):
        fc.kabsch(a, b, w)


def test_structure_superposition(helix):
    moved = helix.copy()
    moved.transform(geo.rotation_about_axis((1, 0, 0), 120.0), (4.0, 5.0, 6.0))
    mob, tgt = fc.paired_coordinates(fc.pair_residues(moved, helix), ("N", "CA", "C", "O"))
    assert len(mob) == 64 and fc.kabsch(mob, tgt).rmsd < 1e-6


def test_identical_structures_score_perfectly(ref):
    model = [geo.add(geo.mat_vec(geo.rotation_about_axis((0.3, 0.1, 1), 77.0), p), (10, -3, 2)) for p in ref]
    assert fc.gdt_ts(model, ref) == pytest.approx(1.0) and fc.gdt_ha(model, ref) == pytest.approx(1.0)
    tm = fc.tm_score(model, ref)
    assert tm.score == pytest.approx(1.0) and tm.aligned_within_5 == 40


def test_scores_degrade_monotonically_with_noise(ref):
    rows = []
    for sigma in (0.3, 1.0, 3.0):
        model = perturbed(ref, sigma, seed=4)
        rows.append((fc.gdt_ts(model, ref), fc.gdt_ha(model, ref), fc.tm_score(model, ref).score))
    for column in zip(*rows):
        assert list(column) == sorted(column, reverse=True)
    assert rows[-1][0] < 0.8 and rows[-1][2] < 0.8


def test_gdt_finds_well_predicted_half(ref):
    model = list(ref[:20]) + [geo.add(p, (0.0, 25.0, 0.0)) for p in ref[20:]]
    result = fc.gdt(model, ref)
    assert result.fractions[1.0] == result.fractions[8.0] == pytest.approx(0.5)
    assert result.score == pytest.approx(0.5) and fc.kabsch(model, ref).rmsd > 5.0
    assert fc.gdt_ts(ref[:20], ref[:20], n_reference=40) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fc.gdt_ts(ref[:20], ref[:20], n_reference=10)


def test_tm_score_scale_and_unrelated_structures(ref):
    assert tm_d0(10) == 0.5 and tm_d0(100) == pytest.approx(1.24 * 85 ** (1 / 3) - 1.8)
    rng, walk = random.Random(1), [(0.0, 0.0, 0.0)]
    for _ in range(39):
        walk.append(geo.add(walk[-1], geo.scale(geo.normalize((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))), 3.8)))
    assert fc.tm_score(walk, ref).score < 0.4


def test_score_validation_and_subset_search(ref):
    with pytest.raises(fc.ComparisonError):
        fc.gdt_ts(ref[:5], ref[:6])
    with pytest.raises(fc.ComparisonError):
        fc.tm_score(ref[:2], ref[:2])
    value, sup = maximal_subset_search(ref, ref, 1.0, lambda d: -max(d))
    assert value == pytest.approx(0.0, abs=1e-6) and sup.rmsd == pytest.approx(0.0, abs=1e-6)
