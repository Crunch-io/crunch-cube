# encoding: utf-8

'''Contains implementation of the Dimension class, for Crunch Cubes.'''

import numpy as np

from .subtotal import Subtotal
from .utils import lazyproperty, memoize
from . import ITEM_DIMENSION_TYPES


class Dimension(object):
    '''Implementation of the Dimension class for Crunch Cubes.

    This class contains all the utility functions for working with
    Crunch Cube dimensions. It also hides some of the internal implementation
    detail from the user, especially for Multiple response variables.
    '''
    def __init__(self, dim, selections=None):
        self._dim = dim
        self._type = self._get_type(dim, selections)

    @lazyproperty
    def values(self):
        values = [
            el.get('numeric_value', np.nan)
            for el in self._elements
            if not el.get('missing')
        ]
        return [val if val is not None else np.nan for val in values]

    @lazyproperty
    def subtotals(self):
        view = self._dim.get('references', {}).get('view', {})

        if not view:
            # View can be both None and {}, thus the edge case.
            return []

        insertions_data = view.get('transform', {}).get('insertions', [])
        subtotals = [Subtotal(data, self) for data in insertions_data]
        return [subtotal for subtotal in subtotals if subtotal.is_valid]

    @classmethod
    def _get_type(cls, dim, selections=None):
        '''Gets the Dimension type.

        MR and CA variables have two subsequent dimension, which are both
        necessary to determine the correct type ('categorical_array', or
        'multiple_response').
        '''
        type_ = dim['type'].get('class')

        if type_:
            if type_ == 'enum':
                if 'subreferences' in dim['references']:
                    return ('multiple_response'
                            if cls._is_multiple_response(selections)
                            else 'categorical_array')
                if 'subtype' in dim['type']:
                    return dim['type']['subtype']['class']

            return type_

        return dim['type']['subtype']['class']

    @classmethod
    def _is_multiple_response(cls, dim):
        if not dim:
            return False

        categories = dim['type'].get('categories')
        if not categories:
            return False

        if len(categories) != 3:
            return False

        mr_ids = (1, 0, -1)
        for i, mr_id in enumerate(mr_ids):
            if categories[i]['id'] != mr_id:
                return False
        return True

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

    @lazyproperty
    def _elements(self):
        if self.type == 'categorical':
            return self._dim['type']['categories']
        return self._dim['type']['elements']

    @lazyproperty
    def inserted_hs_indices(self):
        '''Returns inserted H&S indices for the dimension.'''
        if (self.type in ITEM_DIMENSION_TYPES or not self.subtotals):
            return []  # For CA and MR items, we don't do H&S insertions

        elements = self.elements()
        element_ids = [element['id'] for element in elements]

        top_indexes = []
        middle_indexes = []
        bottom_indexes = []
        for i, st in enumerate(self.subtotals):
            anchor = st.anchor
            if anchor == 'top':
                top_indexes.append(i)
            elif anchor == 'bottom':
                bottom_indexes.append(i)
            else:
                middle_indexes.append(anchor)
        len_top_indexes = len(top_indexes)

        # push all top indexes to the top
        top_indexes = list(range(len_top_indexes))

        # adjust the middle_indexes appropriately
        middle_indexes = [
            i + element_ids.index(index) + len_top_indexes + 1
            for i, index in enumerate(middle_indexes)
        ]

        # what remains is the bottom
        len_non_bottom_indexes = (
            len_top_indexes + len(middle_indexes) + len(elements)
        )
        bottom_indexes = list(range(
            len_non_bottom_indexes, len_non_bottom_indexes + len(bottom_indexes)
        ))

        return top_indexes + middle_indexes + bottom_indexes

    def _transform_anchor(self, subtotal):

        if subtotal.anchor in ['top', 'bottom']:
            return subtotal.anchor

        return self.elements_by_id[subtotal.anchor]['index']

    @lazyproperty
    def elements_by_id(self):
        r = {}
        for i, el in enumerate(self._elements):
            el['index'] = i
            r[el['id']] = el
        return r

    @lazyproperty
    def hs_indices(self):
        '''Headers and Subtotals indices.'''
        if self.is_selections:
            return []

        eid = self.elements_by_id

        indices = []
        for subtotal in self.subtotals:

            inds = []
            for arg in subtotal.args:
                inds.append(eid[arg]['index'])

            indices.append({'anchor_ind': self._transform_anchor(subtotal),
                            'inds': inds})

        # filter where indices aren't available to sum
        indices = [ind for ind in indices if len(ind['inds']) > 0]

        return indices

    @lazyproperty
    def has_transforms(self):
        view = self._dim['references'].get('view')
        if not view:
            return False
        insertions = view.get('transform', {}).get('insertions')
        return insertions is not None

    # API methods

    @lazyproperty
    def name(self):
        '''Name of a cube's dimension.'''
        refs = self._dim['references']
        return refs.get('name', refs.get('alias'))

    @lazyproperty
    def description(self):
        '''Description of a cube's dimension.'''
        refs = self._dim['references']
        return refs.get('description')

    @lazyproperty
    def alias(self):
        '''Alias of a cube's dimension.'''
        refs = self._dim['references']
        return refs.get('alias')

    @lazyproperty
    def type(self):
        '''Get type of the Crunch Dimension.'''
        return self._type

    def labels(self, include_missing=False, include_transforms=False,
               include_cat_ids=False):
        '''Get labels of the Crunch Dimension.'''
        if (not (include_transforms and self.has_transforms) or
                self.type == 'categorical_array'):
            return [
                (
                    self._get_name(el)
                    if not include_cat_ids else
                    (self._get_name(el), el.get('id', -1))
                )
                for (i, el) in enumerate(self._elements)
                if include_missing or i not in self.invalid_indices
            ]

        # Create subtotals names and insert them in labels after
        # appropriate anchors
        labels_with_cat_ids = [{
            'ind': i,
            'id': el['id'],
            'name': self._get_name(el),
        } for (i, el) in enumerate(self._elements)]
        labels_with_cat_ids = self._update_with_subtotals(labels_with_cat_ids)

        valid_indices = self.valid_indices(include_missing)
        return [
            (
                label['name']
                if not include_cat_ids else
                (label['name'], label.get('id', -1))
            )
            for label in labels_with_cat_ids
            if self._include_in_labels(label, valid_indices)
        ]

    def _update_with_subtotals(self, labels_with_cat_ids):
        for subtotal in self.subtotals:
            already_inserted_with_the_same_anchor = [
                index for (index, item) in enumerate(labels_with_cat_ids)
                if 'anchor' in item and item['anchor'] == subtotal.anchor
            ]

            if len(already_inserted_with_the_same_anchor):
                ind_insert = already_inserted_with_the_same_anchor[-1] + 1
            elif subtotal.anchor == 'top':
                ind_insert = 0
            elif subtotal.anchor == 'bottom':
                ind_insert = len(labels_with_cat_ids)
            else:
                ind_insert = next(
                    index for (index, item) in enumerate(labels_with_cat_ids)
                    if item.get('id') == subtotal.anchor
                ) + 1

            labels_with_cat_ids.insert(ind_insert, subtotal.data)

        return labels_with_cat_ids

    @staticmethod
    def _include_in_labels(label_with_ind, valid_indices):
        if label_with_ind.get('ind') is None:
            # In this case, it's a transformation and not an element of the
            # cube. Thus, needs to be included in resulting labels.
            return True

        return label_with_ind['ind'] in valid_indices

    @memoize
    def elements(self, include_missing=False):
        '''Get elements of the crunch Dimension.

        For categorical variables, the elements are represented by categories
        internally. For other variable types, actual 'elements' of the
        Crunch Cube JSON response are returned.
        '''
        if include_missing:
            return self._elements

        return [
            el for (i, el) in enumerate(self._elements)
            if i not in self.invalid_indices
        ]

    @memoize
    def valid_indices(self, include_missing):
        '''Gets valid indices of Crunch Cube Dimension's elements.

        This function needs to be used by CrunchCube class, in order to
        correctly calculate the indices of the result that needs to be
        returned to the user. In most cases, the non-valid indices are
        those of the missing values.
        '''
        if include_missing:
            return range(len(self._elements))
        else:
            return [x for x in range(len(self._elements))
                    if x not in self.invalid_indices]

    @lazyproperty
    def invalid_indices(self):
        return set((
            i for (i, el) in enumerate(self._elements)
            if el.get('missing')
        ))

    @lazyproperty
    def shape(self):
        return len(self._elements)

    @lazyproperty
    def is_selections(self):
        categories = self._elements
        if len(categories) != 3:
            return False

        mr_ids = (1, 0, -1)
        for i, mr_id in enumerate(mr_ids):
            if categories[i]['id'] != mr_id:
                return False
        return True

    def is_mr_selections(self, others):
        '''Check whether a selections dimension has a corresponding items dim

        Sometimes selections are used to in conjunction with another dim,
        that knows where to find them (following it). Other times, they behave
        as a normal categorical dimension. This checks against the aliases of
        all other dims to see which is the case.
        '''
        if self.is_selections:
            for dim in others:
                if dim.alias == self.alias and not dim.is_selections:
                    return True
        return False
