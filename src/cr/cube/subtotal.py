'''Contains implementation of the Subtotal class, for Crunch Cubes.'''

from .utils import lazyproperty


class Subtotal(object):
    '''Implementation of the Insertion class for Crunch Cubes.

    Contains all functionality necessary for retrieving the information
    for subtotals. This functionality is used in the context
    of headers and subtotals.
    '''
    def __init__(self, data, dim):
        self._data = data
        self._dim = dim

    @lazyproperty
    def is_valid(self):
        '''Test if the subtotal data is valid.'''
        if isinstance(self._data, dict):
            required_keys = {'anchor', 'args', 'function', 'name'}
            has_keys = set(self._data.keys()) == required_keys
            if has_keys:
                is_subtotal = self._data['function'] == 'subtotal'
                hs_ids = self._data['args']
                are_hs_ids_valid = self._validate_hs_ids(hs_ids)
                return is_subtotal and are_hs_ids_valid
        return False

    @lazyproperty
    def anchor(self):
        '''Get the anchor of the subtotal (if it's valid).'''
        if not self.is_valid:
            return None

        anchor = self._data['anchor']
        try:
            anchor = int(anchor)
            if anchor not in self._all_dim_ids:
                return 'bottom'
            return anchor
        except (TypeError, ValueError):
            return anchor.lower()

    @lazyproperty
    def _all_dim_ids(self):
        return [el.get('id') for el in self._dim.elements(include_missing=True)]

    def _validate_hs_ids(self, hs_ids):
        element_ids = [el['id'] for el in self._dim.elements()]
        return any(arg in element_ids for arg in hs_ids)

    @lazyproperty
    def args(self):
        '''Get H&S args.'''
        hs_ids = self._data['args']
        if self.is_valid and self._validate_hs_ids(hs_ids):
            return self._data['args']
        return []

    @lazyproperty
    def data(self):
        '''Get data in JSON format.'''
        return self._data
