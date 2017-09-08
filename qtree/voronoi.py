import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi


def _additional_segments(segments, bounds):
    ret = []
    for ib, b in enumerate(bounds):
        for i in range(2):
            wh, rows = np.where(segments[:, :, i] == b[i])
            points = [segment[row] for segment, row in zip(segments[wh], rows)]
            points = np.asarray(points)
            inc_axis = (i + 1) % 2
            lb = b
            ub = [b[i], bounds[(ib + 1) % 2][(i + 1) % 2]]
            if inc_axis == 0:
                ub.reverse()
            points = np.concatenate([[lb], points, [ub]])
            npoints = points[np.argsort(points[:, inc_axis])]
            npoints = points.shape[0]
            lo = points[range(npoints-1), None]
            hi = points[range(1, npoints), None]
            segs = np.concatenate([lo, hi], axis=1)
            ret.append(segs)
    return np.concatenate(ret)


def _clip_edges(verts, bounds):
    for i, vert in enumerate(verts):
        if ((vert[0] < bounds[0][0] or vert[1] < bounds[0][1] or
             vert[0] >= bounds[1][0] or vert[1] >= bounds[1][1])):
            overt = verts[(i + 1) % 2]
            for b1, b2 in _boundary_segments(bounds):
                intersect = _find_intersection(vert, overt, b1, b2)
                if intersect is not None:
                    verts[i] = intersect
                    return np.asarray(verts)
            return None
    return np.asarray(verts)


def _a(obj):
    return np.asarray(obj)


def _boundary_segments(bounds):
    yield bounds[0], _a([bounds[1, 0], bounds[0, 1]])
    yield bounds[0], _a([bounds[0, 0], bounds[1, 1]])
    yield _a([bounds[0, 0], bounds[1, 1]]), bounds[1]
    yield _a([bounds[1, 0], bounds[0, 1]]), bounds[1]


def _find_intersection(p0, p1, p2, p3):
    # see https://stackoverflow.com/questions/563198

    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]

    denom = s10_x * s32_y - s32_x * s10_y

    if denom == 0:
        # collinear
        return None

    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]

    s_numer = s10_x * s02_y - s10_y * s02_x
    t_numer = s32_x * s02_y - s32_y * s02_x

    posd = denom > 0

    if ((posd == (s_numer < 0) or
         posd == (t_numer < 0) or
         posd == (s_numer > denom) or
         posd == (t_numer > denom))):
        # no collision
        return None

    # collision detected

    t = t_numer / denom
    intersection_point = [p0[0] + (t * s10_x), p0[1] + (t * s10_y)]
    return np.asarray(intersection_point)


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

        segments = []
        box_width = bounds[1] - bounds[0]
        center = box_width/2
        for pointidx, simplex in zip(voro.ridge_points, voro.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                segment = _clip_edges(voro.vertices[simplex], bounds)
                if segment is not None:
                    segments.append(segment)
                continue
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex index
            t = voro.points[pointidx[1]] - voro.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = voro.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voro.vertices[i] + direction * box_width
            segment = _clip_edges([voro.vertices[i], far_point], bounds)
            if segment is not None:
                segments.append(segment)
        segments = np.asarray(segments)
        edge_segments = _additional_segments(segments, bounds)
        segments = np.concatenate([segments, edge_segments])
        self.segments = segments

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
