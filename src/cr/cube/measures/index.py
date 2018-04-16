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

        return props / margin

    def _mr_index(self, weighted):
        table = self.cube.table.data(weighted)
        selected = table[self.cube.ind_selected]
        non_selected = table[self.cube.ind_non_selected]

        # mr by mr
        if self.cube.mr_dim_ind == (0, 1) or self.cube.mr_dim_ind == (1, 2):
            row_proportions = self.cube.proportions(axis=0,
                                                    weighted=self.weighted,
                                                    prune=self.prune)
            col_proportions = self.cube.proportions(axis=1,
                                                    weighted=self.weighted,
                                                    prune=self.prune)
            return col_proportions / row_proportions

        # mr by cat
        if self.cube.mr_dim_ind == 0:
            proportions = self.cube.proportions(axis=0,
                                                weighted=weighted,
                                                prune=self.prune)
            margin = np.sum(selected, 1) / np.sum(selected + non_selected, 1)
            return proportions / margin[:, np.newaxis]

        # cat by mr
        if self.cube.mr_dim_ind == 1:
            proportions = self.cube.proportions(axis=1,
                                                weighted=weighted,
                                                prune=self.prune)
            margin = np.sum(selected, 0) / np.sum(selected + non_selected, 0)
            return proportions / margin

        raise ValueError('Unexpected dimension types for cube with MR.')
