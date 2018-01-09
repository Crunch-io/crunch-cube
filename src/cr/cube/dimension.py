'''Contains implementation of the Dimension class, for Crunch Cubes.'''

from cr.cube.subtotal import Subtotal


class Dimension(object):
    '''Implementation of the Dimension class for Crunch Cubes.

    This class contains all the utility functions for working with
    Crunch Cube dimensions. It also hides some of the internal implementation
    detail from the user, especially for Multiple response variables.
    '''
    def __init__(self, dim, selections=None):
        self._dim = dim
        self._type = self._get_type(dim, selections)

    @property
    def subtotals(self):
        insertions_data = self._dim.get('references', {}) \
                .get('view', {}) \
                .get('transform', {}) \
                .get('insertions', [])
        subtotals = [Subtotal(data) for data in insertions_data]
        return [subtotal for subtotal in subtotals if subtotal.is_valid]

    @classmethod
    def _get_type(cls, dim, selections=None):
        '''Gets the Dimension type.

        MR and CA variables have two subsequent dimension, which are both
        necessary to determine the correct type ('categorical_array', or
        'multiple_response').
        '''
        type_ = dim['type'].get('class')

        if type_ and type_ == 'enum' and 'subreferences' in dim['references']:
            return ('multiple_response'
                    if cls._is_multiple_response(selections)
                    else 'categorical_array')

        if type_ and type_ == 'enum' and 'subtype' in dim['type']:
            return dim['type']['subtype']['class']

        if type_:
            return type_

        return dim['type']['subtype']['class']

    @classmethod
    def _is_multiple_response(cls, dim):
        if not dim:
            return False

        categories = dim['type'].get('categories')
        if not categories:
            return False

        ids = [cat['id'] for cat in categories]
        return ids == [1, 0, -1]

    @classmethod
    def _get_name(cls, element):
        name = element.get('name')

        # For categorical variables
        if name:
            return name

        # For numerical, datetime and text variables
        value = element.get('value')
        if value is None:
            return None

        # The following statement is used for compatibility between
        # python 2 and 3. In python 3 everything is 'str' and 'unicode'
        # is not defined. So, if the value is textual, in python 3 the first
        # part of the 'or' statement should short-circuit to 'True'.
        type_ = type(value)
        if type_ == list:
            return '-'.join([str(el) for el in value])
        elif type_ in [float, int]:
            return str(value)
        elif type_ != dict and (type_ == str or type_ == unicode):  # noqa: F821
            return value

        # For categorical array variables
        name = value.get('references', {}).get('name')
        if name:
            return name

        return None

    @property
    def _elements(self):
        if self.type == 'categorical':
            return self._dim['type']['categories']
        return self._dim['type']['elements']

    @property
    def hs_indices(self):
        '''Headers and Subtotals indices.'''
        indices = [{
            'anchor_ind': [
                i
                for (i, el) in enumerate(self._elements)
                if el['id'] == subtotal.anchor
            ][0]
            if (subtotal.anchor in [el['id'] for el in self._elements]) else
            -1,
            'inds': [
                i
                for (i, el) in enumerate(self._elements)
                if el['id'] in subtotal.args
            ],
        } for subtotal in self.subtotals]
        return indices

    @property
    def has_transforms(self):
        view = self._dim['references'].get('view')
        if not view:
            return False
        insertions = view.get('transform', {}).get('insertions')
        return insertions is not None

    # API methods

    @property
    def name(self):
        '''Name of a cube's dimension.'''
        refs = self._dim['references']
        return refs.get('name', refs.get('alias'))

    @property
    def description(self):
        '''Description of a cube's dimension.'''
        refs = self._dim['references']
        return refs.get('description')

    @property
    def alias(self):
        '''Alias of a cube's dimension.'''
        refs = self._dim['references']
        return refs.get('alias')

    @property
    def type(self):
        '''Get type of the Crunch Dimension.'''
        return self._type

    def labels(self, include_missing=False, include_transforms=False,
               include_cat_ids=False):
        '''Get labels of the Crunch Dimension.'''
        valid_indices = self.valid_indices(include_missing)
        if (not (include_transforms and self.has_transforms) or
                self.type == 'categorical_array'):
            return [
                (
                    self._get_name(el)
                    if not include_cat_ids else
                    (self._get_name(el), el.get('id', -1))
                )
                for (i, el) in enumerate(self._elements)
                if i in valid_indices
            ]

        # Create subtotals names and insert them in labels after
        # appropriate anchors
        labels_with_cat_ids = [{
            'ind': i,
            'id': el['id'],
            'name': self._get_name(el),
        } for (i, el) in enumerate(self._elements)]
        for subtotal in self.subtotals:
            # If anchor is 0, the default value of 'next' (which is -1) will
            # add with 1, and amount to the total of 0, which will be the
            # correct index value for the new subtotal.
            ind_insert = next((
                i for (i, x) in enumerate(labels_with_cat_ids)
                if x.get('id') == subtotal.anchor
            ), -1) + 1
            labels_with_cat_ids.insert(ind_insert, subtotal.data)

        return [
            (
                label['name']
                if not include_cat_ids else
                (label['name'], label.get('id', -1))
            )
            for label in labels_with_cat_ids
            if not label.get('ind') or label['ind'] in valid_indices
        ]

    def elements(self, include_missing=False):
        '''Get elements of the crunch Dimension.

        For categorical variables, the elements are represented by categories
        internally. For other variable types, actual 'elements' of the
        Crunch Cube JSON response are returned.
        '''
        valid_indices = self.valid_indices(include_missing)
        return [
            el for (i, el) in enumerate(self._elements)
            if i in valid_indices
        ]

    def valid_indices(self, include_missing):
        '''Gets valid indices of Crunch Cube Dimension's elements.

        This function needs to be used by CrunchCube class, in order to
        correctly calculate the indices of the result that needs to be
        returned to the user. In most cases, the non-valid indices are
        those of the missing values.
        '''
        if include_missing:
            return [i for (i, el) in enumerate(self._elements)]
        else:
            return [i for (i, el) in enumerate(self._elements)
                    if not el.get('missing')]
