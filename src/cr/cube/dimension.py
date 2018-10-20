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

    def __init__(self, dimension_dict, selections=None):
        self._dimension_dict = dimension_dict
        self._selections = selections

    @lazyproperty
    def alias(self):
        """str system (as opposed to human) name for this dimension."""
        refs = self._dimension_dict['references']
        return refs.get('alias')

    @lazyproperty
    def description(self):
        """Description of a cube's dimension."""
        refs = self._dimension_dict['references']
        return refs.get('description')

    @memoize
    def element_indices(self, include_missing):
        """Return tuple of int element idxs for this dimension.

        *include_missing* determines whether missing elements are included or
        only valid element index values are returned.
        """
        return (
            self._all_elements.element_idxs if include_missing else
            self._valid_elements.element_idxs
        )

    @memoize
    def elements(self, include_missing=False):
        """_Elements object providing access to elements of this dimension."""
        return (
            self._all_elements if include_missing else self._valid_elements
        )

    @lazyproperty
    def has_transforms(self):
        view = self._dimension_dict['references'].get('view')
        if not view:
            return False
        insertions = view.get('transform', {}).get('insertions')
        return insertions is not None

    @lazyproperty
    def hs_indices(self):
        """tuple of (anchor_idx, addend_idxs) pair for each subtotal."""
        if self.is_selections:
            return ()

        return tuple(
            (subtotal.anchor_idx, subtotal.addend_idxs)
            for subtotal in self.subtotals
        )

    @lazyproperty
    def inserted_hs_indices(self):
        """Returns inserted H&S indices for the dimension."""
        if (self.type in ITEM_DIMENSION_TYPES or not self.subtotals):
            return []  # For CA and MR items, we don't do H&S insertions

        elements = self._valid_elements
        element_ids = elements.element_ids

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
        """True for the categories dimension of an MR dimension-pair."""
        return (
            len(self._all_elements) == 3 and
            self._all_elements.element_ids == (1, 0, -1)
        )

    def labels(self, include_missing=False, include_transforms=False,
               include_cat_ids=False):
        """Get labels of the Crunch Dimension."""
        if (not (include_transforms and self.has_transforms) or
                self.type == 'categorical_array'):
            elements = (
                self._all_elements if include_missing else
                self._valid_elements
            )
            return [
                (
                    element.name
                    if not include_cat_ids else
                    (element.name, element.element_id)
                )
                for element in elements
            ]

        # Create subtotals names and insert them in labels after
        # appropriate anchors
        labels_with_cat_ids = [{
            'ind': idx,
            'id': element.element_id,
            'name': element.name,
        } for (idx, element) in enumerate(self._all_elements)]
        labels_with_cat_ids = self._update_with_subtotals(labels_with_cat_ids)

        element_indices = self.element_indices(include_missing)
        return [
            (
                label['name']
                if not include_cat_ids else
                (label['name'], label.get('id', -1))
            )
            for label in labels_with_cat_ids
            if self._include_in_labels(label, element_indices)
        ]

    @lazyproperty
    def name(self):
        """Name of a cube's dimension."""
        refs = self._dimension_dict['references']
        return refs.get('name', refs.get('alias'))

    @lazyproperty
    def numeric_values(self):
        """tuple of numeric values for valid elements of this dimension.

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
        return tuple(
            element.numeric_value for element in self._valid_elements
        )

    @lazyproperty
    def shape(self):
        return len(self._all_elements)

    @lazyproperty
    def subtotals(self):
        """_Subtotals sequence object for this dimension.

        The subtotals sequence provides access to any subtotal insertions
        defined on this dimension.
        """
        view = self._dimension_dict.get('references', {}).get('view', {})
        # ---view can be both None and {}, thus the edge case.---
        insertion_dicts = (
            [] if view is None else
            view.get('transform', {}).get('insertions', [])
        )
        return _Subtotals(insertion_dicts, self._valid_elements)

    @lazyproperty
    def type(self):
        """Get type of the Crunch Dimension."""
        return self.__class__._get_type(self._dimension_dict, self._selections)

    @lazyproperty
    def _all_elements(self):
        """_AllElements object providing cats or subvars of this dimension."""
        return _AllElements(self._dimension_dict['type'])

    @classmethod
    def _get_type(cls, dimension_dict, selections=None):
        """Gets the Dimension type.

        MR and CA variables have two subsequent dimension, which are both
        necessary to determine the correct type ('categorical_array', or
        'multiple_response').
        """
        type_ = dimension_dict['type'].get('class')

        if type_:
            if type_ == 'enum':
                if 'subreferences' in dimension_dict['references']:
                    return ('multiple_response'
                            if cls._is_multiple_response(selections)
                            else 'categorical_array')
                if 'subtype' in dimension_dict['type']:
                    return dimension_dict['type']['subtype']['class']

            return type_

        return dimension_dict['type']['subtype']['class']

    @staticmethod
    def _include_in_labels(label_with_ind, valid_indices):
        if label_with_ind.get('ind') is None:
            # In this case, it's a transformation and not an element of the
            # cube. Thus, needs to be included in resulting labels.
            return True

        return label_with_ind['ind'] in valid_indices

    @classmethod
    def _is_multiple_response(cls, dimension_dict):
        if not dimension_dict:
            return False

        categories = dimension_dict['type'].get('categories')
        if not categories:
            return False

        if len(categories) != 3:
            return False

        mr_ids = (1, 0, -1)
        for i, mr_id in enumerate(mr_ids):
            if categories[i]['id'] != mr_id:
                return False
        return True

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

    @lazyproperty
    def _valid_elements(self):
        """_Elements object providing access to non-missing elements.

        Any categories or subvariables representing missing data are excluded
        from the collection; this sequence represents a subset of that
        provided by `._all_elements`.
        """
        return self._all_elements.valid_elements


class _BaseElements(Sequence):
    """Base class for element sequence containers."""

    def __init__(self, type_dict):
        self._type_dict = type_dict

    def __getitem__(self, idx_or_slice):
        """Implements indexed access."""
        return self._elements[idx_or_slice]

    def __iter__(self):
        """Implements (efficient) iterability."""
        return iter(self._elements)

    def __len__(self):
        """Implements len(elements)."""
        return len(self._elements)

    @lazyproperty
    def element_ids(self):
        """tuple of element-id for each element in collection.

        Element ids appear in the order they occur in the cube response.
        """
        return tuple(element.element_id for element in self._elements)

    @lazyproperty
    def element_idxs(self):
        """tuple of element-index for each element in collection.

        Element index values represent the position of this element in the
        dimension-dict it came from. In the case of an _AllElements object,
        it will simply be a tuple(range(len(all_elements))).
        """
        return tuple(element.index for element in self._elements)

    def get_by_id(self, element_id):
        """Return _Element object identified by *element_id*.

        Raises KeyError if not found. Only elements known to this collection
        are accessible for lookup. For example, a _ValidElements object will
        raise KeyError for the id of a missing element.
        """
        return self._elements_by_id[element_id]

    @lazyproperty
    def _elements(self):
        """tuple storing actual sequence of element objects.

        Must be implemented by each subclass.
        """
        raise NotImplementedError('must be implemented by each subclass')

    @lazyproperty
    def _element_makings(self):
        """(ElementCls, element_dicts) pair for this dimension's elements.

        All the elements of a given dimension are the same type. This method
        determines the type (class) and source dicts for the elements of this
        dimension and provides them for the element factory.
        """
        if self._type_dict['class'] == 'categorical':
            return _Category, self._type_dict['categories']
        return _Element, self._type_dict['elements']

    @lazyproperty
    def _elements_by_id(self):
        """dict mapping each element by its id."""
        return {element.element_id: element for element in self._elements}


class _AllElements(_BaseElements):
    """Sequence of _BaseElement subclass objects for a dimension.

    Each element is either a category or a subvariable.
    """

    @lazyproperty
    def valid_elements(self):
        """_ValidElements object containing only non-missing elements."""
        return _ValidElements(self._elements)

    @lazyproperty
    def _elements(self):
        """Composed tuple storing actual sequence of element objects."""
        ElementCls, element_dicts = self._element_makings
        return tuple(
            ElementCls(element_dict, idx)
            for idx, element_dict in enumerate(element_dicts)
        )


class _ValidElements(_BaseElements):
    """Sequence of non-missing element objects for a dimension.

    *type_dict* is the dict on the 'type': key of a dimension and
    *all_elements* is an instance of _AllElements containing all the elements
    of a dimension. This object is only intended to be constructed by
    _AllElements.valid_elements and there should be no reason to construct it
    directly.
    """

    def __init__(self, all_elements):
        self._all_elements = all_elements

    @lazyproperty
    def _elements(self):
        """tuple containing actual sequence of element objects."""
        return tuple(
            element for element in self._all_elements
            if not element.missing
        )


class _BaseElement(object):
    """Base class for element objects."""

    def __init__(self, element_dict, index):
        self._element_dict = element_dict
        self._index = index

    @lazyproperty
    def element_id(self):
        """int identifier for this category or subvariable."""
        return self._element_dict['id']

    @lazyproperty
    def index(self):
        """int offset at which this element appears in elements sequence."""
        return self._index

    @lazyproperty
    def missing(self):
        """True if this element represents missing data.

        False if this category or subvariable represents valid (collected)
        data.
        """
        return bool(self._element_dict.get('missing'))

    @lazyproperty
    def numeric_value(self):
        """Numeric value assigned to element by user, np.nan if absent."""
        numeric_value = self._element_dict.get('numeric_value')
        return np.nan if numeric_value is None else numeric_value


class _Category(_BaseElement):
    """A category on a categorical dimension."""

    def __init__(self, category_dict, index):
        super(_Category, self).__init__(category_dict, index)
        self._category_dict = category_dict

    @lazyproperty
    def name(self):
        """str name assigned to this category or subvariable by user."""
        name = self._category_dict.get('name')
        return name if name else ''


class _Element(_BaseElement):
    """A subvariable on an MR or CA enum dimension."""

    @lazyproperty
    def name(self):
        """str display-name for this element, '' when name is absent.

        This property handles numeric, datetime and text variables, but also
        subvar dimensions
        """
        value = self._element_dict.get('value')
        type_name = type(value).__name__

        if type_name == 'NoneType':
            return ''

        if type_name == 'list':
            # ---like '10-15' or 'A-F'---
            return '-'.join([str(item) for item in value])

        if type_name in ('float', 'int'):
            return str(value)

        if type_name in ('str', 'unicode'):
            return value

        # ---For CA and MR subvar dimensions---
        name = value.get('references', {}).get('name')
        return name if name else ''


class _Subtotals(Sequence):
    """Sequence of _Subtotal objects for a dimension.

    Each _Subtotal object represents a "subtotal" insertion transformation
    defined for the dimension.

    A subtotal can only involve valid (i.e. non-missing) elements.
    """

    def __init__(self, insertion_dicts, valid_elements):
        self._insertion_dicts = insertion_dicts
        self._valid_elements = valid_elements

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
    def _element_ids(self):
        """frozenset of int id of each non-missing cat or subvar in dim."""
        return frozenset(self._valid_elements.element_ids)

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
            if not self._element_ids.intersection(insertion_dict['args']):
                continue

            # ---an insertion-dict that successfully runs this gauntlet
            # ---is a valid subtotal dict
            yield insertion_dict

    @lazyproperty
    def _subtotals(self):
        """Composed tuple storing actual sequence of _Subtotal objects."""
        return tuple(
            _Subtotal(subtotal_dict, self._valid_elements)
            for subtotal_dict in self._iter_valid_subtotal_dicts()
        )


class _Subtotal(object):
    """A subtotal insertion on a cube dimension."""

    def __init__(self, subtotal_dict, valid_elements):
        self._subtotal_dict = subtotal_dict
        self._valid_elements = valid_elements

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
            if anchor not in self._valid_elements.element_ids:
                return 'bottom'
            return anchor
        except (TypeError, ValueError):
            return anchor.lower()

    @lazyproperty
    def anchor_idx(self):
        """int or str representing index of anchor element in dimension.

        When the anchor is an operation, like 'top' or 'bottom'
        """
        anchor = self.anchor
        if anchor in ['top', 'bottom']:
            return anchor
        return self._valid_elements.get_by_id(anchor).index

    @lazyproperty
    def addend_ids(self):
        """tuple of int ids of elements contributing to this subtotal.

        Any element id not present in the dimension or present but
        representing missing data is excluded.
        """
        return tuple(
            arg for arg in self._subtotal_dict.get('args', [])
            if arg in self._valid_elements.element_ids
        )

    @lazyproperty
    def addend_idxs(self):
        """tuple of int index of each addend element for this subtotal.

        The length of the tuple is the same as that for `.addend_ids`, but
        each value repesents the offset of that element within the dimension,
        rather than its element id.
        """
        return tuple(
            self._valid_elements.get_by_id(addend_id).index
            for addend_id in self.addend_ids
        )

    @lazyproperty
    def label_dict(self):
        """dict having 'name' and 'anchor' items for this subtotal."""
        return {
            'anchor': self.anchor,
            'name': self.name
        }

    @lazyproperty
    def name(self):
        """str display name for this subtotal, suitable for use as label."""
        name = self._subtotal_dict.get('name')
        return name if name else ''
