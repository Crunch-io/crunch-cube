'''Contains implementation of the Index service class.'''

from __future__ import division

import numpy as np


class Index(object):
    '''Implementation of the Index service class for Crunch Cubes.'''
    def __init__(self, cube, weighted, prune):
        self._cube = cube
        self._weighted = weighted
        self._prune = prune

    @property
    def cube(self):
        '''Get stored cube object.'''
        return self._cube

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
            return self._mr_index(self.weighted)

        margin = (
            self.cube.margin(axis=0, weighted=self.weighted, prune=self.prune) /
            self.cube.margin(weighted=self.weighted, prune=self.prune)
        )
        props = self.cube.proportions(axis=1, weighted=self.weighted, prune=self.prune)

        res = props / margin
        if isinstance(res, np.ma.core.MaskedArray):
            res.mask = props.mask
        return res
        # return props / margin

    def _mr_index(self, weighted):
        table = self.cube.table.data(weighted)
        selected = table[self.cube.ind_selected]
        non_selected = table[self.cube.ind_non_selected]

        if self.cube.mr_dim_ind == 0 or self.cube.mr_dim_ind == (0, 1):
            margin = np.sum(selected, 1) / np.sum(selected + non_selected, 1)
            return (self.cube.proportions(axis=0, weighted=weighted) /
                    margin[:, np.newaxis])

        if self.cube.mr_dim_ind == 1:
            margin = np.sum(selected, 0) / np.sum(selected + non_selected, 0)
            return self.cube.proportions(axis=1, weighted=weighted) / margin

        raise ValueError('Unexpected dimension types for cube with MR.')
