'''Contains implementation of the Index service class.'''

from __future__ import division

import numpy as np


class Index(object):
    '''Implementation of the Index service class for Crunch Cubes.'''
    def __init__(self, cube, axis, weighted, prune):
        self._cube = cube
        self._axis = axis
        self._weighted = weighted
        self._prune = prune

    @property
    def cube(self):
        '''Get stored cube object.'''
        return self._cube

    @property
    def axis(self):
        '''Get axis property.'''
        return self._axis

    @property
    def weighted(self):
        '''Get weighted property.'''
        return self._weighted

    @property
    def prune(self):
        '''Get prune property.'''
        return self._prune

    @property
    def data(self):
        '''Return table index by margin.'''
        if self.cube.has_mr:
            return self._mr_index(self.axis, self.weighted)

        margin = (
            self.cube.margin(
                axis=self.axis, weighted=self.weighted, prune=self.prune
            ) /
            self.cube.margin(weighted=self.weighted, prune=self.prune)
        )
        props = self.cube.proportions(
            axis=(1 - self.axis),
            weighted=self.weighted,
            prune=self.prune,
        )

        if self.axis == 1:
            margin = margin[:, np.newaxis]

        return props / margin

    def _mr_index(self, axis, weighted):
        cube = self.cube

        table = cube.table.data(weighted)
        selected = table[cube.ind_selected]
        non_selected = table[cube.ind_non_selected]

        if cube.mr_dim_ind == 0 or cube.mr_dim_ind == (0, 1):
            if axis != 0:
                # MR x CAT index table only defined for column direction.
                return np.full(cube.as_array().shape, np.nan)
            margin = np.sum(selected, 1) / np.sum(selected + non_selected, 1)
            return (cube.proportions(axis=axis, weighted=weighted) /
                    margin[:, np.newaxis])

        if cube.mr_dim_ind == 1:
            if axis == 0:
                # CAT x MR index table not defined for column direction.
                return np.full(cube.as_array().shape, np.nan)
            margin = np.sum(selected, 0) / np.sum(selected + non_selected, 0)
            return cube.proportions(weighted=weighted) / margin

        raise ValueError('Unexpected dimension types for cube with MR.')
