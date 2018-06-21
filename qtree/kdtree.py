
import numpy as np

from cykdtree import PyKDTree

DIRECTION_MAPPING = {
    'x': 0,
    'y': 1,
    'z': 2,
}


def _plot(plot_nodes, filename, outlines_only=False):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    if outlines_only:
        cb_pad = 0
        figsize = (4, 4)
    else:
        cb_pad = .06
        figsize = (4.5, 4)
    fig = plt.figure(figsize=figsize)
    aspect = figsize[1]/figsize[0]
    pad = .02
    axs = (1 - 2*pad)
    axes = fig.add_axes([pad, pad, axs*aspect, axs])
    if outlines_only is False:
        cb_ax = fig.add_axes([axs*aspect, pad, axs - cb_pad - axs*aspect, axs])
    axes.set_aspect('equal')
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)

    patches = []
    npts_a = []

    for le, re, npts in plot_nodes:
        patches.append(Rectangle(le, re[0] - le[0], re[1] - le[1]))
        npts_a.append(npts)

    pc = PatchCollection(patches, edgecolors='k')
    if outlines_only is False:
        pc.set_array(np.array(npts_a))
        pc.set_clim(0, 16)
    else:
        pc.set_facecolor('red')
        pc.set_alpha(0.1)
    axes.add_collection(pc)

    if outlines_only is False:
        cbar = fig.colorbar(pc, ax=axes, cax=cb_ax, cmap='viridis')
        cbar.set_label('Number of Points')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


class ParticleProjectionKDTree(object):

    def __init__(self, positions, direction, left_edge, right_edge,
                 amr_nested=True):
        """A data structure for projecting particles using a KDTree

        Parameters
        ----------
        positions : ndarray
            The input positions to project. The positions must be of
            shape ``(nparticles, 3)``.
        direction : string
            The axis to project along. One of 'x', 'y', or 'z'.
        left_edge : ndarray
            The 3D coordinates of the lower left bounding box corner.
            Particles to the left of this point will be discarded.
        right_edge : ndarray
            The 3D coordinates of the upper right bounding box corner.
            Particles to the left of this point will be discarded.
        amr_nested : bool
            Force the kdtree to split at nearest AMR cell boundary for
            KDTree level
        """
        self.direction = direction
        self.kdtree = PyKDTree(positions, left_edge, right_edge,
                               periodic=(True, True, True), leafsize=16,
                               amr_nested=amr_nested)

    def plot(self, filename=None):

        plot_nodes = []

        for leaf in self.kdtree.leaves:
            le = leaf.left_edge
            re = leaf.right_edge
            d = DIRECTION_MAPPING[self.direction]
            le = np.hstack([le[:d], le[d+1:]])
            re = np.hstack([re[:d], re[d+1:]])
            plot_nodes.append((le, re, leaf.npts))

        _plot(plot_nodes, filename, outlines_only=True)


class ParticleSliceKDTree(object):

    def __init__(self, positions, direction, coord, left_edge, right_edge,
                 amr_nested=True):
        """A data structure for projecting particles using a KDTree

        Parameters
        ----------
        positions : ndarray
            The input positions to project. The positions must be of
            shape ``(nparticles, 3)``.
        direction : string
            The axis to project along. One of 'x', 'y', or 'z'.
        coord : float
            The coordinate along the ``direction`` axis to slice through.
        left_edge : ndarray
            The 3D coordinates of the lower left bounding box corner.
            Particles to the left of this point will be discarded.
        right_edge : ndarray
            The 3D coordinates of the upper right bounding box corner.
            Particles to the left of this point will be discarded.
        amr_nested : bool
            Force the kdtree to split at nearest AMR cell boundary for
            KDTree level
        """
        self.coord = coord
        self.direction = direction
        self.kdtree = PyKDTree(positions, left_edge, right_edge,
                               periodic=(True, True, True), leafsize=16,
                               amr_nested=amr_nested)

    def plot(self, filename):

        plot_nodes = []

        for leaf in self.kdtree.leaves:
            le = leaf.left_edge
            re = leaf.right_edge
            d = DIRECTION_MAPPING[self.direction]
            if not le[d] < self.coord <= re[d]:
                continue
            le = np.hstack([le[:d], le[d+1:]])
            re = np.hstack([re[:d], re[d+1:]])
            plot_nodes.append((le, re, leaf.npts))

        _plot(plot_nodes, filename)


def natural_pow2_resolution(vals):
    """Find power of 2 that offers a natural spacing given value"""
    return 2**(np.floor(np.log2(min(vals))) - 5)


def pin_power_2(inp, res, top_or_bottom):
    """Round inp to the nearest power of 2"""
    ret = [0, 0]
    for i, val in enumerate(inp):
        if val == 0:
            ret[i] = val
            continue
        # find nearest power of 2
        adj = np.fmod(val, res)
        ret[i] = val - (-(1 + top_or_bottom)*res/2 + adj)
        if ret[i] > 1:
            ret[i] = 1
    return ret
