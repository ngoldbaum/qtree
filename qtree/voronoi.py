import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi

from qtree.utils import _points_in_poly


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

    def pixelize(self, image):
        """pixelize the deposit_field onto an image

        Parameters
        ----------
        image : 2D array
            Image to pixelize onto
        """
        image = np.asarray(image)

        if len(image.shape) != 2:
            raise RuntimeError("Must pixelize onto 2D image")

        voro = self.voro
        regions = voro.regions
        bounds = self.bounds

        dx = 1/image.shape[0]
        dy = 1/image.shape[1]

        xb = bounds[1, 0] - bounds[0, 0] - dx
        yb = bounds[1, 1] - bounds[0, 1] - dy

        xlin = np.arange(image.shape[0])/(image.shape[0] - 1) * xb + dx/2
        ylin = np.arange(image.shape[1])/(image.shape[1] - 1) * yb + dy/2

        x, y = np.meshgrid(xlin, ylin)

        for i, point_coord in enumerate(voro.points):
            region_idx = voro.point_region[i]
            region = regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            vertices = voro.vertices[region]
            vx = vertices[:, 0]
            vy = vertices[:, 1]
            area = 0.5*np.abs(np.dot(vx, np.roll(vy, 1)) -
                              np.dot(vy, np.roll(vx, 1)))
            in_poly = _points_in_poly(vertices, x, y)
            image[np.where(in_poly)] = self.deposit_field[i] / area

        return image

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
