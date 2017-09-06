try:
    import enum
except ImportError:
    # need enum34 on python2.7
    import enum34 as enum

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


@enum.unique
class _Direction(enum.Enum):
    SOUTHWEST = (0, 0)
    SOUTHEAST = (1, 0)
    NORTHWEST = (0, 1)
    NORTHEAST = (1, 1)


_NODE_CAPACITY = 4

_offsets = {
    _Direction.SOUTHWEST: np.array((-1, -1)),
    _Direction.SOUTHEAST: np.array((1, -1)),
    _Direction.NORTHWEST: np.array((-1, 1)),
    _Direction.NORTHEAST: np.array((1, 1)),
}


class ParticleQuadTreeNode(object):
    __slots__ = ('positions', 'num_particles', 'center', 'half_width',
                 'northeast', 'northwest', 'southeast', 'southwest',
                 'deposit_field', '_left_edge', '_right_edge')

    def __init__(self, center, half_width):
        """A QuadTree data structure containing particles

        Parameters
        ----------
        center : 2-element iterable
            The center of the quad tree node
        half_width : float
            The half-width of the node. Currently only square nodes are
            supported.
        """
        self.positions = np.empty((_NODE_CAPACITY, 2))
        self.deposit_field = np.empty(_NODE_CAPACITY)
        self.positions[:] = np.nan
        self.num_particles = 0

        center = np.array(center)

        if center.shape != (2,):
            raise RuntimeError(
                "Received center with shape %s but expected (2,)"
                % (center.shape,))

        self.center = center
        self.half_width = half_width

        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None
        self._left_edge = None
        self._right_edge = None

    def insert(self, positions, deposit_field=None):
        """Insert particles into the quadtree

        Parameters
        ----------
        positions : 2 element iterable or iterable of 2-element iterables
            Positions of the particles to be inserted.
        deposit_field : iterable, optional
            Field to be deposited and pixelized. Must have the same number of
            elements as the number of positions.
        """
        positions = np.asarray(positions)
        if len(positions.shape) == 1:
            positions = np.asarray([positions])

        nparticles = positions.shape[0]

        if deposit_field is not None and nparticles != deposit_field.shape[0]:
            raise RuntimeError(
                "Received %s deposit_field entries but received %s particle "
                "positions" % (deposit_field.shape[0], nparticles))

        if positions.shape[-1] != 2:
            raise RuntimeError(
                "Received %sD positions %s but expected 2D positions"
                % (positions.shape,))

        # check if particle is inside this node
        if not ((positions > self.left_edge).all() and
                (positions < self.right_edge).all()):
            raise RuntimeError(
                "positions outside node with left_edge=%s and right_edge=%s"
                % (self.left_edge, self.right_edge))

        cur_np = self.num_particles
        self.num_particles += nparticles

        if self.num_particles <= _NODE_CAPACITY:
            self.positions[cur_np: cur_np + nparticles] = positions
            if deposit_field is not None:
                self.deposit_field[cur_np: cur_np + nparticles] = deposit_field
            return

        if self.is_leaf:
            positions = np.vstack((self.positions[:cur_np], positions))
            if deposit_field is not None:
                deposit_field = np.concatenate(
                    (self.deposit_field[:cur_np], deposit_field))
            self.positions = None
            self.deposit_field = None

        # adding another particle requires refining the tree
        center = self.center
        for direction in _Direction:
            pos_gt_cen = (positions > center).astype(int)
            inds = (pos_gt_cen == direction.value).all(axis=-1).nonzero()[0]
            if deposit_field is not None:
                dep_field = deposit_field[inds]
            else:
                dep_field = None
            self._insert_child(direction, positions[inds], dep_field)

    def _insert_child(self, direction, positions, deposit_field):
        child_name = direction.name.lower()
        child_node = getattr(self, child_name)
        if child_node is None:
            offset = _offsets[direction]
            child_node = ParticleQuadTreeNode(
                self.center + self.half_width/2 * offset, self.half_width/2)
            setattr(self, child_name, child_node)
        child_node.insert(positions, deposit_field)

    @property
    def children(self):
        for d in _Direction:
            child = getattr(self, d.name.lower())
            if child is not None:
                yield child

    @property
    def leaves(self):
        if self.is_leaf:
            yield self
        for child in self.children:
            for leaf in child.leaves:
                yield leaf

    @property
    def area(self):
        return (self.right_edge - self.left_edge).prod()

    @property
    def left_edge(self):
        if self._left_edge is None:
            self._left_edge = self.center - self.half_width
        return self._left_edge

    @property
    def right_edge(self):
        if self._right_edge is None:
            self._right_edge = self.center + self.half_width
        return self._right_edge

    @property
    def is_leaf(self):
        return self.positions is not None

    def _plot_subtree(self, fig, axes):
        patch = Rectangle(self.left_edge, self.half_width*2, self.half_width*2,
                          fill=False)
        axes.add_patch(patch)

        if self.is_leaf:
            positions = self.positions[:self.num_particles]
            axes.scatter(positions[:, 0], positions[:, 1], .1, color='k')

        for child in self.children:
            child._plot_subtree(fig, axes)

    def plot(self, filename=None, fig=None, axes=None):
        """Plot this quadtree node and its subtree"""
        fig = plt.figure(figsize=(4, 4))
        axes = fig.add_axes([.01, .01, .98, .98])
        axes.set_aspect('equal')
        plt.axis('off')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        self._plot_subtree(fig, axes)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
