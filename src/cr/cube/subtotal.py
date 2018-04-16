'''Contains implementation of the Subtotal class, for Crunch Cubes.'''


class Subtotal(object):
    '''Implementation of the Insertion class for Crunch Cubes.

    Contains all functionality necessary for retrieving the information
    for subtotals. This functionality is used in the context
    of headers and subtotals.
    '''
    def __init__(self, data, dim):
        self._data = data
        self._dim = dim

    @property
    def is_valid(self):
        '''Test if the subtotal data is valid.'''
        if isinstance(self._data, dict):
            required_keys = {'anchor', 'args', 'function', 'name'}
            has_keys = set(self._data.keys()) == required_keys
            if has_keys:
                return self._data['function'] == 'subtotal'
        return False

    @property
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

    @property
    def _all_dim_ids(self):
        return [el.get('id') for el in self._dim.elements(include_missing=True)]

    @property
    def args(self):
        '''Get H&S args.'''
        if self.is_valid:
            return self._data['args']
        return []

    @property
    def data(self):
        '''Get data in JSON format.'''
        return self._data
