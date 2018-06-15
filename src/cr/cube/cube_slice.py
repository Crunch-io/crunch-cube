'''Home of the CubeSlice class.'''

import numpy as np


# pylint: disable=too-few-public-methods
class CubeSlice(object):
    '''Implementation of CubeSlice class.

    The CubeSlice is used to uniformly represent all cubes as 2D tables.
    For 1D cubes, this is achieved by inflating. For 3D cubes, this is
    achieved by slicing.
    '''

    def __init__(self, cube, index):
        self._cube = cube
        self._index = index

    def __getattr__(self, attr):
        cube_attr = getattr(self._cube, attr)

        if self._cube.ndim == 3 and hasattr(cube_attr, '__len__'):
            return cube_attr[-2:]

        return cube_attr

    def __getattribute__(self, attr):

        cube_methods = [
            'as_array', 'margin', 'population_counts', 'proportions', 'index',
            'zscore', 'pvals', 'prune_indices', 'labels', 'inserted_hs_indices',
        ]

        class CubeCaller(object):
            '''Class used for self._cube method calls when not defined.'''
            def __init__(self, cube):
                self._cube = cube

            # pylint: disable=protected-access
            def __call__(self, *args, **kwargs):
                return self._cube._call_cube_method(attr, *args, **kwargs)

        if attr in cube_methods:
            return CubeCaller(self)

        return object.__getattribute__(self, attr)

    def _update_args(self, kwargs):

        if self.ndim < 3:
            # If cube is 2D it doesn't actually have slices (itself is a slice).
            # In this case we don't need to convert any arguments, but just
            # pass them to the underlying cube (which is the slice).
            return kwargs

        axis = kwargs.get('axis')
        if axis is None or isinstance(axis, tuple):
            # If no axis was passed, we don't need to update anything. If axis
            # was a tuple, it means that the caller was expecting a 3D cube.
            return kwargs

        kwargs['axis'] += 1
        return kwargs

    def _update_result(self, result):
        if self.ndim < 3 or len(result) - 1 < self._index:
            return result
        result = result[self._index]
        if isinstance(result, tuple):
            return list(result)
        if not isinstance(result, np.ndarray):
            result = np.array([result])
        return result

    def _call_cube_method(self, method, *args, **kwargs):
        kwargs = self._update_args(kwargs)
        result = getattr(self._cube, method)(*args, **kwargs)
        if method in ['labels', 'inserted_hs_indices']:
            return result[-2:]
        return self._update_result(result)

    @property
    def name(self):
        '''Get slice name.

        In case of 2D return cube name. In case of 3D, return the combination
        of the cube name with the label of the corresponding slice
        (nth label of the 0th dimension).
        '''
        title = self._cube.name

        if self.ndim < 3:
            return title

        table_name = self._cube.labels()[0][self._index]
        return '%s: %s' % (title, table_name)
