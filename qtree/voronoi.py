import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi


class ParticleVoronoiMesh(object):

    def __init__(self, positions, deposit_field, bounds):
        """A voronoi mesh generated from particle positions

        Parameters
        ----------
        positions : iterable of 2-element iterables
            Positions of the particles to be inserted.
        deposit_field : iterable, optional
            Field to be deposited and pixelized. Must have the same
            number of elements as the number of positions.
        bounds : 2-element iterable of two-tuples
            The coordinates of the lower-left and upper-right corners
            of the bounds of the mesh. Particles outside the bounds are
            discarded.
        """
        positions = np.asarray(positions)
        self.num_particles = nparticles = positions.shape[0]
        self.bounds = bounds = np.asarray(bounds)

        if deposit_field is not None and nparticles != deposit_field.shape[0]:
            raise RuntimeError(
                "Received %s deposit_field entries but received %s particle "
                "positions" % (deposit_field.shape[0], nparticles))

        if positions.shape[-1] != 2:
            raise RuntimeError(
                "Received %sD positions %s but expected 2D positions"
                % (positions.shape,))

        self.voro = voro = Voronoi(positions)
        self.deposit_field = deposit_field

        ridge_verts = np.array(voro.ridge_vertices)
        ridge_verts = ridge_verts[(ridge_verts != -1).all(axis=-1)]
        self.segments = voro.vertices[ridge_verts]

    def plot(self, filename=None):
        """Plot the mesh"""
        fig = plt.figure(figsize=(4, 4))
        axes = fig.add_axes([.01, .01, .98, .98])
        axes.set_aspect('equal')
        plt.axis('off')
        lc = LineCollection(self.segments, color='k', linewidths=0.5)
        axes.add_collection(lc)
        positions = self.voro.points
        axes.scatter(positions[:, 0], positions[:, 1], s=.2, color='k',
                     marker='o')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=400)
