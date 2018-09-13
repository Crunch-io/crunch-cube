'''Grouping of various cube methods'''

import numpy as np

from ..utils import lazyproperty
from ..dimension import Dimension


class DataTable(object):
    '''Groups together useful cube utility methods.'''
    def __init__(self, cube):
        self._cube = cube

    # Properties

    @lazyproperty
    def all_dimensions(self):
        '''Gets the dimensions of the crunch cube.

        This function is internal, and is not mean to be used by ouside users
        of the CrunchCube class. The main reason for this is the internal
        representation of the different variable types (namely the MR and the
        CA). These types have two dimensions each, but in the case of MR, the
        second dimensions shouldn't be visible to the user. This function
        returns such dimensions as well, since they're necessary for the
        correct implementation of the functionality for the MR type.
        The version that is mentioned to be used by users is the
        property 'dimensions'.
        '''
        entries = self._cube['result']['dimensions']
        return [
            (
                # Multiple Response and Categorical Array variables have
                # two subsequent dimensions (elements and selections). For
                # this reason it's necessary to pass in both of them in the
                # Dimension class init method. This is needed in order to
                # determine the correct type (CA or MR). We only skip the
                # two-argument constructor for the last dimension in the list
                # (where it's not possible to fetch the subsequent one).
                Dimension(entry)
                if i + 1 >= len(entries)
                else Dimension(entry, entries[i + 1])
            )
            for (i, entry) in enumerate(entries)
        ]

    @lazyproperty
    def mr_selections_indices(self):
        '''Gets indices of each 'selection' dim, for corresponding MR dim.

        Multiple Response (MR) and Categorical Array (CA) variables are
        represented by two dimensions each. These dimensions can be thought of
        as 'elements' and 'selections'. This function returns the indices of
        the 'selections' dimension for each MR variable.
        '''
        mr_dimensions_indices = [
            i for (i, dim) in enumerate(self.all_dimensions)
            if (i + 1 < len(self.all_dimensions) and
                dim.type == 'multiple_response')
        ]

        # For each MR and CA dimension, the 'selections' dimension
        # follows right after it (in the originating cube).
        # Here we increase the MR index by 1, which gives us
        # the index of the corresponding 'selections' dimension.
        return [i + 1 for i in mr_dimensions_indices]

    @lazyproperty
    def has_means(self):
        '''Check if cube has means.'''
        measures = self._cube.get('result', {}).get('measures')
        if not measures:
            return False
        return measures.get('mean', None) is not None

    @lazyproperty
    def is_weighted(self):
        '''Check if the cube dataset is weighted.'''
        weighted = self._cube.get('query', {}).get('weight', None) is not None
        weighted = weighted or self._cube.get('weight_var', None) is not None
        weighted = weighted or self._cube.get('weight_url', None) is not None
        weighted = weighted or (
            self._cube['result']['counts'] !=
            self._cube['result']['measures'].get('count', {}).get('data')
        )
        return weighted

    @lazyproperty
    def missing(self):
        '''Get missing count of a cube.'''
        if self.has_means:
            return self._cube['result']['measures']['mean']['n_missing']
        return self._cube['result'].get('missing')

    @lazyproperty
    def filter_annotation(self):
        '''Get cube's filter annotation.'''
        return self._cube.get('filter_names', [])

    # API Methods

    def count(self, weighted=True):
        '''Get cube's count with automatic weighted/unweighted selection.'''
        if weighted and self.is_weighted:
            return sum(
                self._cube['result']['measures'].get('count', {}).get('data')
            )
        return self._cube['result']['n']

    def flat_values(self, weighted, margin=False):
        '''Gets the flat values from the original cube response.

        Params
            weighted (bool): Whether to get the unweighted or weighted counts
            margin (bool): If we're doing the calculations for the margin, we
                don't want any other measure (e.g. means), but only counts
                (which may be weighted or unweighted, depending on the type
                of the margin).
        Returns
            values (ndarray): The flattened array, which represents the result
                of the cube computation.
        '''
        values = self._cube['result']['counts']
        if self.has_means and not margin:
            mean = self._cube['result']['measures'].get('mean', {})
            values = mean.get('data', values)
        elif weighted and self.is_weighted:
            count = self._cube['result']['measures'].get('count', {})
            values = count.get('data', values)
        values = [(val if not type(val) is dict else np.nan)
                  for val in values]
        return values

    @lazyproperty
    def _shape(self):
        return tuple([dim.shape for dim in self.all_dimensions])

    @lazyproperty
    def counts(self):
        unfiltered = self._cube['result'].get('unfiltered')
        filtered = self._cube['result'].get('filtered')
        return unfiltered, filtered

    def data(self, weighted, margin=False):
        '''Get the data in non-flattened shape.

        Converts the flattened shape (original response) into non-flattened
        shape (count of elements per cube dimension). E.g. for a CAT x CAT
        cube, with 2 categories in each dimension (variable), we end up with
        a ndarray of shape (2, 2).
        '''
        values = self.flat_values(weighted, margin)
        return np.array(values).reshape(self._shape)
