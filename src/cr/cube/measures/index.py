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
            return self._mr_index()
        margin = (
            self.cube.margin(axis=0, weighted=self.weighted, prune=self.prune) /
            self.cube.margin(weighted=self.weighted, prune=self.prune)
        )
        proportions = self.cube.proportions(
            axis=1, weighted=self.weighted, prune=self.prune
        )
        return proportions / margin

    def _mr_index(self):
        # mr by mr
        if len(self.cube.dimensions) == 2 and self.cube.mr_dim_ind == (0, 1):
            col_proportions = self.cube.proportions(axis=0,
                                                    weighted=self.weighted,
                                                    prune=self.prune)
            row_proportions = self.cube.proportions(axis=1,
                                                    weighted=self.weighted,
                                                    prune=self.prune)
            return col_proportions / row_proportions

        # mr by cat and cat by mr
        if self.cube.mr_dim_ind == 0 or self.cube.mr_dim_ind == 1:
            axis = self.cube.mr_dim_ind
            margin = (
                self.cube.margin(axis=1 - axis, weighted=self.weighted, prune=self.prune) /
                self.cube.margin(weighted=self.weighted, prune=self.prune)
            )
            proportions = self.cube.proportions(axis=axis, weighted=self.weighted, prune=self.prune)
            if self.cube.mr_dim_ind == 0:
                margin = margin[:, np.newaxis]  # pivot
            return proportions / margin

        raise ValueError('Unexpected dimension types for cube with MR.')
