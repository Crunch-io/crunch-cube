# encoding: utf-8

"""Provides the Dimension class."""

from collections import Sequence

import numpy as np

from cr.cube import ITEM_DIMENSION_TYPES
from cr.cube.util import lazyproperty, memoize


class Dimension(object):
    """Represents one dimension of a cube response.

    Each dimension represents one of the variables in a cube response. For
    example, a query to cross-tabulate snack-food preference against region
    will have two variables (snack-food preference and region) and will produce
    a two-dimensional (2D) cube response. That cube will have two of these
    dimension objects, which are accessed using
    :attr:`.CrunchCube.dimensions`.
    """

    def __init__(self, dim, selections=None):
        self._dim = dim
        self._type = self._get_type(dim, selections)

    @lazyproperty
    def alias(self):
        """str system (as opposed to human) name for this dimension."""
        refs = self._dim['references']
        return refs.get('alias')

    @lazyproperty
    def description(self):
        """Description of a cube's dimension."""
        refs = self._dim['references']
        return refs.get('description')

    @memoize
    def elements(self, include_missing=False):
        """Get elements of the crunch Dimension.

        For categorical variables, the elements are represented by categories
        internally. For other variable types, actual 'elements' of the
        Crunch Cube JSON response are returned.
        """
        if include_missing:
            return self._elements

        return [
            el for (i, el) in enumerate(self._elements)
            if i not in self.invalid_indices
        ]

    @lazyproperty
    def elements_by_id(self):
        r = {}
        for i, el in enumerate(self._elements):
            el['index'] = i
            r[el['id']] = el
        return r

    @lazyproperty
    def has_transforms(self):
        view = self._dim['references'].get('view')
        if not view:
            return False
        insertions = view.get('transform', {}).get('insertions')
        return insertions is not None

    @lazyproperty
    def hs_indices(self):
        """list of dict each having 'anchor_ind' and 'inds' items.

        Represents subtotal insertions in terms of their anchor and element
        indices.
        """
        # TODO: This implementation is an example of primitive obsession,
        # using a primitive type (dict in this case) to do the job of an
        # object. This increases complexity and risk. There is no place to
        # document the contracts of the object's methods and its form and
        # behaviors need to be learned by each new developer the hard way.
        # Also, it's mutable and vulnerable to unintended state changes for
        # that reason. I'm inclined to think this behavior should be migrated
        # to _Subtotals such that subtotal element indices can be retrieved
        # directly from the _Subtotal object.
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
    def inserted_hs_indices(self):
        """Returns inserted H&S indices for the dimension."""
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

    @lazyproperty
    def invalid_indices(self):
        return set((
            i for (i, el) in enumerate(self._elements)
            if el.get('missing')
        ))

    def is_mr_selections(self, others):
        """Return True if this dimension var is multiple-response selections.

        *others* is an iterable containing all dimensions in the cube.

        Sometimes selections are used in conjunction with another dimension
        that knows where to find them (following it). Other times, they
        behave as a normal categorical dimension. This checks against the
        aliases of all other dims to see which is the case.
        """
        if self.is_selections:
            for dim in others:
                if dim.alias == self.alias and not dim.is_selections:
                    return True
        return False

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

    def labels(self, include_missing=False, include_transforms=False,
               include_cat_ids=False):
        """Get labels of the Crunch Dimension."""
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

    @lazyproperty
    def name(self):
        """Name of a cube's dimension."""
        refs = self._dim['references']
        return refs.get('name', refs.get('alias'))

    @lazyproperty
    def shape(self):
        return len(self._elements)

    @lazyproperty
    def subtotals(self):
        """_Subtotals sequence object for this dimension.

        The subtotals sequence provides access to any subtotal insertions
        defined on this dimension.
        """
        view = self._dim.get('references', {}).get('view', {})
        # ---view can be both None and {}, thus the edge case.---
        insertion_dicts = (
            [] if view is None else
            view.get('transform', {}).get('insertions', [])
        )
        return _Subtotals(insertion_dicts, self)

    @lazyproperty
    def type(self):
        """Get type of the Crunch Dimension."""
        return self._type

    @memoize
    def valid_indices(self, include_missing):
        """Gets valid indices of Crunch Cube Dimension's elements.

        This function needs to be used by CrunchCube class, in order to
        correctly calculate the indices of the result that needs to be
        returned to the user. In most cases, the non-valid indices are
        those of the missing values.
        """
        if include_missing:
            return range(len(self._elements))
        else:
            return [x for x in range(len(self._elements))
                    if x not in self.invalid_indices]

    @lazyproperty
    def values(self):
        """list of numeric values for elements of this dimension.

        Each category of a categorical variable can be assigned a *numeric
        value*. For example, one might assign `like=1, dislike=-1,
        neutral=0`. These numeric mappings allow quantitative operations
        (such as mean) to be applied to what now forms a *scale* (in this
        example, a scale of preference).

        The numeric values appear in the same order as the
        categories/elements of this dimension. Each element is represented by
        a value, but an element with no numeric value appears as `np.nan` in
        the returned list.
        """
        values = [
            el.get('numeric_value', np.nan)
            for el in self._elements
            if not el.get('missing')
        ]
        return [val if val is not None else np.nan for val in values]

    @lazyproperty
    def _elements(self):
        if self.type == 'categorical':
            return self._dim['type']['categories']
        return self._dim['type']['elements']

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

    @classmethod
    def _get_type(cls, dim, selections=None):
        """Gets the Dimension type.

        MR and CA variables have two subsequent dimension, which are both
        necessary to determine the correct type ('categorical_array', or
        'multiple_response').
        """
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

    @staticmethod
    def _include_in_labels(label_with_ind, valid_indices):
        if label_with_ind.get('ind') is None:
            # In this case, it's a transformation and not an element of the
            # cube. Thus, needs to be included in resulting labels.
            return True

        return label_with_ind['ind'] in valid_indices

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

    def _transform_anchor(self, subtotal):

        if subtotal.anchor in ['top', 'bottom']:
            return subtotal.anchor

        return self.elements_by_id[subtotal.anchor]['index']

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

            labels_with_cat_ids.insert(ind_insert, subtotal.label_dict)

        return labels_with_cat_ids


class _Subtotals(Sequence):
    """Sequence of _Subtotal objects for a dimension.

    Each _Subtotal object represents a "subtotal" insertion transformation
    defined for the dimension.

    A subtotal can only involve valid (i.e. non-missing) elements.
    """

    def __init__(self, insertion_dicts, dimension):
        self._insertion_dicts = insertion_dicts
        self._dimension = dimension

    def __getitem__(self, idx_or_slice):
        """Implements indexed access."""
        return self._subtotals[idx_or_slice]

    def __iter__(self):
        """Implements (efficient) iterability."""
        return iter(self._subtotals)

    def __len__(self):
        """Implements len(subtotals)."""
        return len(self._subtotals)

    @lazyproperty
    def _dim_element_ids(self):
        """frozenset of int id of each cat or subvar in dimension.

        Ids of categories or subvariables representing missing values are not
        included.
        """
        return frozenset(
            element.get('id')
            for element in self._dimension.elements()
        )

    def _iter_valid_subtotal_dicts(self):
        """Generate each insertion dict that represents a valid subtotal."""
        for insertion_dict in self._insertion_dicts:
            # ---skip any non-dicts---
            if not isinstance(insertion_dict, dict):
                continue

            # ---skip any non-subtotal insertions---
            if insertion_dict.get('function') != 'subtotal':
                continue

            # ---skip any malformed subtotal-dicts---
            if not {'anchor', 'args', 'name'}.issubset(insertion_dict.keys()):
                continue

            # ---skip if doesn't reference at least one non-missing element---
            if not self._dim_element_ids.intersection(insertion_dict['args']):
                continue

            # ---an insertion-dict that successfully runs this gauntlet
            # ---is a valid subtotal dict
            yield insertion_dict

    @lazyproperty
    def _subtotals(self):
        """Composed tuple storing actual sequence of _Subtotal objects."""
        return tuple(
            _Subtotal(subtotal_dict, self._dimension)
            for subtotal_dict in self._iter_valid_subtotal_dicts()
        )


class _Subtotal(object):
    """Implementation of the Insertion class for Crunch Cubes.

    Contains all functionality necessary for retrieving the information
    for subtotals. This functionality is used in the context
    of headers and subtotals.
    """

    def __init__(self, subtotal_dict, dimension):
        self._subtotal_dict = subtotal_dict
        self._dimension = dimension

    @lazyproperty
    def anchor(self):
        """int or str indicating element under which to insert this subtotal.

        An int anchor is the id of the dimension element (category or
        subvariable) under which to place this subtotal. The return value can
        also be one of 'top' or 'bottom'.

        The return value defaults to 'bottom' for an anchor referring to an
        element that is no longer present in the dimension or an element that
        represents missing data.
        """
        anchor = self._subtotal_dict['anchor']
        try:
            anchor = int(anchor)
            if anchor not in self._valid_dim_element_ids:
                return 'bottom'
            return anchor
        except (TypeError, ValueError):
            return anchor.lower()

    @lazyproperty
    def args(self):
        """tuple of int ids of elements contributing to this subtotal.

        Any element id not present in the dimension or present but
        representing missing data is excluded.
        """
        return tuple(
            arg for arg in self._subtotal_dict.get('args', [])
            if arg in self._valid_dim_element_ids
        )

    @lazyproperty
    def label_dict(self):
        """dict having 'name' and 'anchor' items for this subtotal."""
        return {key: self._subtotal_dict[key] for key in ('anchor', 'name')}

    @lazyproperty
    def _valid_dim_element_ids(self):
        """frozenset of int id of each valid cat or subvar in dimension.

        The term "valid" here means "not-missing", so intuitively, that it
        represents "valid" data actually collected, as opposed to
        characterizing why the data is missing.
        """
        return frozenset(
            element['id']
            for element in self._dimension.elements(include_missing=False)
            if 'id' in element
        )
