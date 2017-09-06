import numpy as np
from qtree import ParticleQuadTreeNode


def test_quadtree():
    np.random.seed(0x4d3d3d3)
    input_npart = 1000
    positions = np.random.normal(loc=0.5, scale=0.1, size=(input_npart, 2))
    positions = np.clip(positions, 0, 1.0)

    tree = ParticleQuadTreeNode([0.5, 0.5], 0.5)

    tree.insert(positions)

    npart = 0

    for leaf in tree.leaves:
        npart += leaf.num_particles

    assert npart == input_npart


def test_quadtree_deposit_field():
    np.random.seed(0x4d3d3d3)
    input_npart = 1000
    positions = np.random.normal(loc=0.5, scale=0.1, size=(input_npart, 2))
    positions = np.clip(positions, 0, 1.0)

    masses = np.ones(input_npart)

    tree = ParticleQuadTreeNode([0.5, 0.5], 0.5)

    tree.insert(positions, masses)

    mass = 0

    for leaf in tree.leaves:
        mass += leaf.deposit_field[:leaf.num_particles].sum()

    assert mass == input_npart


def test_leaf_area():
    np.random.seed(0x4d3d3d3)
    input_npart = 1000
    positions = np.random.normal(loc=0.5, scale=0.1, size=(input_npart, 2))
    positions = np.clip(positions, 0, 1.0)

    tree = ParticleQuadTreeNode([0.5, 0.5], 0.5)

    tree.insert(positions)

    area = 0

    for leaf in tree.leaves:
        area += leaf.area

    assert area == 1
