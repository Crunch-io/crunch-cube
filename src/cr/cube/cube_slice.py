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

        class CubeCaller(object):
            '''Class used for self._cube method calls when not defined.'''
            def __init__(self, cube_slice):
                self._cube_slice = cube_slice

            # pylint: disable=protected-access
            def __call__(self, *args, **kwargs):
                return self._cube_slice._call_cube_method(attr, *args, **kwargs)

        # API Method calls
        if callable(cube_attr):
            return CubeCaller(self)

        # API properties
        get_only_last_two = (
            self._cube.ndim == 3 and
            hasattr(cube_attr, '__len__') and
            attr != 'name'
        )
        if get_only_last_two:
            return cube_attr[-2:]

        # If not defined on self._cube, return CubeSlice properties
        return cube_attr

    def _update_args(self, kwargs):

        if self.ndim < 3:
            # If cube is 2D it doesn't actually have slices (itself is a slice).
            # In this case we don't need to convert any arguments, but just
            # pass them to the underlying cube (which is the slice).
            return kwargs

        axis = kwargs.get('axis')
        if isinstance(axis, int):
            kwargs['axis'] += 1

        hs_dims_key = (
            'transforms'
            if 'transforms' in kwargs else
            'include_transforms_for_dims'
        )
        hs_dims = kwargs.get(hs_dims_key)
        if isinstance(hs_dims, list):
            kwargs[hs_dims_key] = [dim + 1 for dim in hs_dims]

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
    def table_name(self):
        '''Get slice name.

        In case of 2D return cube name. In case of 3D, return the combination
        of the cube name with the label of the corresponding slice
        (nth label of the 0th dimension).
        '''
        if self.ndim < 3:
            return None

        title = self._cube.name
        table_name = self._cube.labels()[0][self._index]
        return '%s: %s' % (title, table_name)

    @property
    def row_dim_ind(self):
        """
        Index of the row dimension in the cube
        :rtype: int
        """
        return 0

    @property
    def col_dim_ind(self):
        """
        Index of the column dimension in the cube
        :rtype: int
        """
        return 1

    @property
    def has_ca(self):
        return 'categorical_array' in self.dim_types

    @property
    def mr_dim_ind(self):
        mr_dim_ind = self._cube.mr_dim_ind
        if self.ndim == 3:
            if isinstance(mr_dim_ind, int):
                return mr_dim_ind - 1
            elif isinstance(mr_dim_ind, tuple):
                return tuple([i - 1 for i in mr_dim_ind])

        return mr_dim_ind

    @property
    def ca_main_axis(self):
        '''For univariate CA, the main axis is the categorical axis'''
        ca_ind = self.dim_types.index('categorical_array')
        if ca_ind is not None:
            return 1 - ca_ind
        else:
            return None
