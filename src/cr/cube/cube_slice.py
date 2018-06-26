'''Home of the CubeSlice class.'''

from functools import partial
import numpy as np


# pylint: disable=too-few-public-methods
class CubeSlice(object):
    '''Implementation of CubeSlice class.

    The CubeSlice is used to uniformly represent all cubes as 2D tables.
    For 1D cubes, this is achieved by inflating. For 3D cubes, this is
    achieved by slicing.
    '''

    row_dim_ind = 0
    col_dim_ind = 1

    def __init__(self, cube, index):
        self._cube = cube
        self._index = index

    def __getattr__(self, attr):
        cube_attr = getattr(self._cube, attr)

        # API Method calls
        if callable(cube_attr):
            return partial(self._call_cube_method, attr)

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

        # Handling API methods that include 'axis' parameter

        axis = kwargs.get('axis')
        # Expected usage of the 'axis' parameter from CubeSlice is 0, 1, or
        # None. CrunchCube handles all other logic. The only 'smart' thing
        # about the handling here, is that the axes are increased for 3D cubes.
        # This way the 3Dness is hidden from the user and he still sees 2D
        # crosstabs, with col and row axes (0 and 1), which are transformed to
        # corresponding numbers in case of 3D cubes (namely 1 and 2). In the
        # case of None, we need to analyze across all valid dimensions, and the
        # CrunchCube takes care of that (no need to update axis if it's None).
        # If the user provides a tuple, it's considered that he "knows" what
        # he's doing, and the axis argument is not updated in this case.
        if isinstance(axis, int):
            kwargs['axis'] += 1

        # Handling API methods that include H&S parameter

        # For most cr.cube methods, we use the 'include_transforms_for_dims'
        # parameter name. For some, namely the prune_indices, we use the
        # 'transforms'. These are parameters that tell to the cr.cube "which
        # dimensions to include the H&S for". The only point of this parameter
        # (from the perspective of the cr.exporter) is to exclude the 0th
        # dimension's H&S in the case of 3D cubes.
        hs_dims_key = (
            'transforms'
            if 'transforms' in kwargs else
            'include_transforms_for_dims'
        )
        hs_dims = kwargs.get(hs_dims_key)
        if isinstance(hs_dims, list):
            # Keep the 2D illusion for the user. If a user sees a 2D slice, he
            # still needs to be able to address both dimensions (for which he
            # wants the H&S included) as 0 and 1. Since these are offset by a 0
            # dimension in a 3D case, inside the cr.cube, we need to increase
            # the indexes of the required dims.
            kwargs[hs_dims_key] = [dim + 1 for dim in hs_dims]

        return kwargs

    def _update_result(self, result):
        if self.ndim < 3 or len(result) - 1 < self._index:
            return result
        result = result[self._index]
        if isinstance(result, tuple):
            return np.array(result)
        elif not isinstance(result, np.ndarray):
            result = np.array([result])
        return result

    def _call_cube_method(self, method, *args, **kwargs):
        kwargs = self._update_args(kwargs)
        result = getattr(self._cube, method)(*args, **kwargs)
        if method in ('labels', 'inserted_hs_indices'):
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
    def has_ca(self):
        '''Check if the cube slice has the CA dimension.

        This is used to distinguish between slices that are considered 'normal'
        (like CAT x CAT), that might be a part of te 3D cube that has 0th dim
        as the CA items (subvars). In such a case, we still need to process
        the slices 'normally', and not address the CA items constraints.
        '''
        return 'categorical_array' in self.dim_types

    @property
    def mr_dim_ind(self):
        '''Get the correct index of the MR dimension in the cube slice.'''
        mr_dim_ind = self._cube.mr_dim_ind
        if self.ndim == 3:
            if isinstance(mr_dim_ind, int):
                return mr_dim_ind - 1
            elif isinstance(mr_dim_ind, tuple):
                return tuple(i - 1 for i in mr_dim_ind)

        return mr_dim_ind

    @property
    def ca_main_axis(self):
        '''For univariate CA, the main axis is the categorical axis'''
        ca_ind = self.dim_types.index('categorical_array')
        if ca_ind is not None:
            return 1 - ca_ind
        else:
            return None

    def labels(self, hs_dims=None, prune=False):
        '''Get labels for the cube slice, and perform pruning by slice.'''
        labels = self._cube.labels(include_transforms_for_dims=hs_dims)[-2:]
        if not prune:
            return labels

        def prune_dimension_labels(labels, prune_indices):
            '''Get pruned labels for single dimension, besed on prune inds.'''
            labels = [
                label for label, prune in zip(labels, prune_indices)
                if not prune
            ]
            return labels

        labels = [
            prune_dimension_labels(dim_labels, dim_prune_inds)
            for dim_labels, dim_prune_inds in
            zip(labels, self.prune_indices(transforms=hs_dims))
        ]
        return labels

    @property
    def has_mr(self):
        '''True if the slice has MR dimension.

        This property needs to be overridden, because we don't care about the
        0th dimension (and if it's an MR) in the case of a 3D cube.
        '''
        return 'multiple_response' in self.dim_types
