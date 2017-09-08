import numpy as np
import tempfile

from qtree.voronoi import ParticleVoronoiMesh


def test_voronoi():
    np.random.seed(0x4d3d3d3)
    input_npart = 1000
    positions = np.random.normal(loc=0.5, scale=0.1, size=(input_npart, 2))
    positions = np.clip(positions, 0, 1.0)

    masses = np.ones(input_npart)

    mesh = ParticleVoronoiMesh(positions, masses, [[0, 0], [1, 1]])

    with tempfile.NamedTemporaryFile() as fp:
        mesh.plot(fp.name)
