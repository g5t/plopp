# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from ..core.limits import find_limits, fix_empty_range
from ..core.utils import value_to_string

import pythreejs as p3
import numpy as np
from matplotlib import ticker
import scipp as sc


def _get_delta(x, axis):
    return (x[axis][1] - x[axis][0]).value


def _get_offsets(limits, axis, ind):
    offsets = np.array([limits[i][ind].value for i in range(3)])
    offsets[axis] = 0
    return offsets


def _make_geometry(limits):
    return p3.EdgesGeometry(
        p3.BoxBufferGeometry(width=_get_delta(limits, axis=0),
                             height=_get_delta(limits, axis=1),
                             depth=_get_delta(limits, axis=2)))


def _make_sprite(string, position, color="black", size=1.0):
    """
    Make a text-based sprite for axis tick
    """
    sm = p3.SpriteMaterial(map=p3.TextTexture(string=string,
                                              color=color,
                                              size=300,
                                              squareTexture=True),
                           transparent=True)
    return p3.Sprite(material=sm, position=position, scale=[size, size, size])


class Outline(p3.Group):

    def __init__(self, limits=None, tick_size=None):
        """
        Make a point cloud using pythreejs
        """
        self.box = None
        self.ticks = None
        self.labels = None

        if limits is None:
            limits = [sc.array(dims=[dim], values=[0, 1]) for dim in 'xyz']

        self.tick_size = tick_size
        if self.tick_size is None:
            self.tick_size = 0.05 * np.mean(
                [_get_delta(limits, axis=i) for i in range(3)])

        super().__init__()

        self.update(limits=limits)
        # # self.aadd = p3.Group()
        # for obj in (self.box, self.ticks, self.labels):
        #     self.add(obj)

    def _make_ticks(self, limits, center):
        """
        Create ticklabels on outline edges
        """
        ticks_group = p3.Group()
        iden = np.identity(3, dtype=np.float32)
        ticker_ = ticker.MaxNLocator(5)
        for axis in range(3):
            ticks = ticker_.tick_values(limits[axis][0].value, limits[axis][1].value)
            for tick in ticks:
                if limits[axis][0].value <= tick <= limits[axis][1].value:
                    tick_pos = iden[axis] * tick + _get_offsets(limits, axis, 0)
                    ticks_group.add(
                        _make_sprite(string=value_to_string(tick, precision=1),
                                     position=tick_pos.tolist(),
                                     size=self.tick_size))
        return ticks_group

    def _make_labels(self, limits, center):
        """
        Create ticklabels on outline edges
        """
        labels_group = p3.Group()
        # iden = np.identity(3, dtype=np.float32)
        for axis in range(3):
            axis_label = f'{limits[axis].dim} [{limits[axis].unit}]'
            # Offset labels 5% beyond axis ticks to reduce overlap
            delta = 0.05
            labels_group.add(
                _make_sprite(string=axis_label,
                             position=(np.roll([1, 0, 0], axis) * center[axis] +
                                       (1.0 + delta) * _get_offsets(limits, axis, 0) -
                                       delta * _get_offsets(limits, axis, 1)).tolist(),
                             size=self.tick_size * 0.3 * len(axis_label)))

        return labels_group

    def update(self, limits):
        center = [var.mean().value for var in limits]
        self.add(
            p3.LineSegments(geometry=_make_geometry(limits),
                            material=p3.LineBasicMaterial(color='#000000'),
                            position=center))
        self.add(self._make_ticks(limits=limits, center=center))
        self.add(self._make_labels(limits=limits, center=center))

    @property
    def visible(self):
        return self.all.visible

    @visible.setter
    def visible(self, val):
        self.all.visible = val
