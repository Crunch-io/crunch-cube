# encoding: utf-8

"""Provides the Index function object class."""

from __future__ import division

import numpy as np


class Index(object):
    """Function object providing index calculations for a CrunchCube object.

    It's only interface is its `.data()` classmethod. It is not intended to
    be instantiated directly.
    """

    @classmethod
    def data(cls, cube, weighted, prune):
        """Return ndarray representing table index by margin."""
        return cls()._data(cube, weighted, prune)

    def _data(self, cube, weighted, prune):
        """ndarray representing table index by margin."""
        result = []
        for slice_ in cube.slices:
            if cube.has_mr:
                return self._mr_index(cube, weighted, prune)
            num = slice_.margin(axis=0, weighted=weighted, prune=prune)
            den = slice_.margin(weighted=weighted, prune=prune)
            margin = num / den
            proportions = slice_.proportions(axis=1, weighted=weighted, prune=prune)
            result.append(proportions / margin)

        if len(result) == 1 and cube.ndim < 3:
            result = result[0]
        else:
            if prune:
                mask = np.array([slice_.mask for slice_ in result])
                result = np.ma.masked_array(result, mask)
            else:
                result = np.array(result)

        return result

    def _mr_index(self, cube, weighted, prune):
        # mr by mr
        if len(cube.dimensions) == 2 and cube.mr_dim_ind == (0, 1):
            col_proportions = cube.proportions(axis=0, weighted=weighted, prune=prune)
            row_proportions = cube.proportions(axis=1, weighted=weighted, prune=prune)
            return col_proportions / row_proportions

        # mr by cat and cat by mr
        if cube.mr_dim_ind == 0 or cube.mr_dim_ind == 1:
            axis = cube.mr_dim_ind
            num = cube.margin(axis=1 - axis, weighted=weighted, prune=prune)
            den = cube.margin(weighted=weighted, prune=prune)
            margin = num / den
            proportions = cube.proportions(axis=axis, weighted=weighted, prune=prune)
            if cube.mr_dim_ind == 0:
                margin = margin[:, np.newaxis]  # pivot
            return proportions / margin

        raise ValueError("Unexpected dimension types for cube with MR.")
