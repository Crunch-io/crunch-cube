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

    def __init__(self, dimension_dict, next_dimension_dict=None):
        self._dimension_dict = dimension_dict
        self._next_dimension_dict = next_dimension_dict

    @lazyproperty
    def alias(self):
        """str system (as opposed to human) name for this dimension."""
        refs = self._dimension_dict['references']
        return refs.get('alias')

    @lazyproperty
    def description(self):
        """str description of this dimension."""
        description = self._dimension_dict['references'].get('description')
        return description if description else ''

    @lazyproperty
    def dimension_type(self):
        """str representing type of this cube dimension."""
        # ---all this logic really belongs in the Dimensions collection
        # ---object, which is where it will move to once that's implemented

        def next_dim_is_mr_cat():
            """True if subsequent dimension is an MR_CAT dimension."""
            if not self._next_dimension_dict:
                return False

            categories = self._next_dimension_dict['type'].get('categories')
            if not categories:
                return False

            return (
                [category.get('id') for category in categories] == [1, 0, -1]
            )

        type_dict = self._dimension_dict['type']
        type_class = type_dict.get('class')

        if not type_class:
            # ---numeric and text are like this---
            return type_dict['subtype']['class']

        if type_class == 'enum':
            if 'subreferences' in self._dimension_dict['references']:
                return (
                    'multiple_response' if next_dim_is_mr_cat()
                    else 'categorical_array'
                )
            if 'subtype' in type_dict:
                # ---datetime is like this (enum without subreferences)---
                return type_dict['subtype']['class']

        return type_class

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
        """True if there are subtotals on this dimension, False otherwise."""
        return len(self._subtotals) > 0

    @lazyproperty
    def hs_indices(self):
        """tuple of (anchor_idx, addend_idxs) pair for each subtotal.

        Example::

            (
                (2, (0, 1, 2)),
                (3, (3,)),
                ('bottom', (4, 5))
            )

        Note that the `anchor_idx` item in the first position of each pair
        can be 'top' or 'bottom' as well as an int. The `addend_idxs` tuple
        will always contains at least one index (a subtotal with no addends
        is ignored).
        """
        if self.is_selections:
            return ()

        return tuple(
            (subtotal.anchor_idx, subtotal.addend_idxs)
            for subtotal in self._subtotals
        )

    @lazyproperty
    def inserted_hs_indices(self):
        """list of int index of each inserted subtotal for the dimension.

        Each value represents the position of a subtotal in the interleaved
        sequence of elements and subtotals items.
        """
        # ---don't do H&S insertions for CA and MR subvar dimensions---
        if self.dimension_type in ITEM_DIMENSION_TYPES:
            return []

        return [
            idx for idx, item
            in enumerate(self._iter_interleaved_items(self._valid_elements))
            if item.is_insertion
        ]

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
        """Return list of str labels for the elements of this dimension.

        Returns a list of (label, element_id) pairs if *include_cat_ids* is
        True. The `element_id` value in the second position of the pair is
        None for subtotal items (which don't have an element-id).
        """
        # TODO: Having an alternate return type triggered by a flag-parameter
        # (`include_cat_ids` in this case) is poor practice. Using flags like
        # that effectively squashes what should be two methods into one.
        # Either get rid of the need for that alternate return value type or
        # create a separate method for it.
        elements = (
            self._all_elements if include_missing else self._valid_elements
        )

        include_subtotals = (
            include_transforms and
            self.dimension_type != 'categorical_array'
        )

        # ---items are elements or subtotals, interleaved in display order---
        interleaved_items = tuple(self._iter_interleaved_items(elements))

        labels = list(
            item.label
            for item in interleaved_items
            if include_subtotals or not item.is_insertion
        )

        if include_cat_ids:
            element_ids = tuple(
                None if item.is_insertion else item.element_id
                for item in interleaved_items
                if include_subtotals or not item.is_insertion
            )
            return list(zip(labels, element_ids))

        return labels

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
    def _all_elements(self):
        """_AllElements object providing cats or subvars of this dimension."""
        return _AllElements(self._dimension_dict['type'])

    def _iter_interleaved_items(self, elements):
        """Generate element or subtotal items in interleaved order.

        This ordering corresponds to how value "rows" (or columns) are to
        appear after subtotals have been inserted at their anchor locations.
        Where more than one subtotal is anchored to the same location, they
        appear in their document order in the cube response.

        Only elements in the passed *elements* collection appear, which
        allows control over whether missing elements are included by choosing
        `._all_elements` or `._valid_elements`.
        """
        subtotals = self._subtotals

        for subtotal in subtotals.iter_for_anchor('top'):
            yield subtotal

        for element in elements:
            yield element
            for subtotal in subtotals.iter_for_anchor(element.element_id):
                yield subtotal

        for subtotal in subtotals.iter_for_anchor('bottom'):
            yield subtotal

    @lazyproperty
    def _subtotals(self):
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
        """int offset at which this element appears in dimension.

        This position is based upon the document position of this element in
        the cube response. No adjustment for missing elements is made.
        """
        return self._index

    @property
    def is_insertion(self):
        """True if this item represents an insertion (e.g. subtotal).

        Unconditionally False for all element types.
        """
        return False

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
    def label(self):
        """str display name assigned to this category by user."""
        name = self._category_dict.get('name')
        return name if name else ''


class _Element(_BaseElement):
    """A subvariable on an MR or CA enum dimension."""

    @lazyproperty
    def label(self):
        """str display-name for this element, '' when absent from cube response.

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

    def iter_for_anchor(self, anchor):
        """Generate each subtotal having matching *anchor*."""
        return (
            subtotal for subtotal in self._subtotals
            if subtotal.anchor == anchor
        )

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

    @property
    def is_insertion(self):
        """True if this item represents an insertion (e.g. subtotal).

        Unconditionally True for _Subtotal objects.
        """
        return True

    @lazyproperty
    def label(self):
        """str display name for this subtotal, suitable for use as label."""
        name = self._subtotal_dict.get('name')
        return name if name else ''
