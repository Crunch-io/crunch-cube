# encoding: utf-8

"""Provides the Dimension class."""

from collections import Sequence

import numpy as np

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class _BaseDimensions(Sequence):
    """Base class for dimension collections."""

    def __getitem__(self, idx_or_slice):
        """Implements indexed access."""
        return self._dimensions[idx_or_slice]

    def __iter__(self):
        """Implements (efficient) iterability."""
        return iter(self._dimensions)

    def __len__(self):
        """Implements len(elements)."""
        return len(self._dimensions)

    @lazyproperty
    def _dimensions(self):
        """tuple of dimension objects in this collection.

        This composed tuple is the source for the dimension objects in this
        collection.
        """
        raise NotImplementedError("must be implemented by each sublass")


class AllDimensions(_BaseDimensions):
    """Collection containing every dimension defined in cube response."""

    def __init__(self, dimension_dicts):
        self._dimension_dicts = dimension_dicts

    @lazyproperty
    def apparent_dimensions(self):
        """_ApparentDimensions collection of the "visible" dimensions.

        The two dimensions for a multiple-response (MR) variable are
        conflated into a single dimensions in this collection.
        """
        return _ApparentDimensions(all_dimensions=self._dimensions)

    @lazyproperty
    def shape(self):
        """Tuple of int element count for each dimension.

        This corresponds to the shape of the ndarray representing the raw
        cube response values (raw meaning including missing and prunable
        elements and any MR_CAT dimensions).
        """
        return tuple(d.shape for d in self._dimensions)

    @lazyproperty
    def _dimensions(self):
        """tuple of dimension objects in this collection.

        This composed tuple is the internal source for the dimension objects
        in this collection.
        """
        return tuple(_DimensionFactory.iter_dimensions(self._dimension_dicts))


class _ApparentDimensions(_BaseDimensions):
    """Collection containing only "user" dimensions of a cube."""

    def __init__(self, all_dimensions):
        self._all_dimensions = all_dimensions

    @lazyproperty
    def _dimensions(self):
        """tuple of dimension objects in this collection.

        This composed tuple is the source for the dimension objects in this
        collection.
        """
        return tuple(d for d in self._all_dimensions if d.dimension_type != DT.MR_CAT)


class _DimensionFactory(object):
    """Produce Dimension objects of correct type from dimension-dicts.

    "type" here is primarily the `.dimension_type` value of the dimension,
    although if `Dimension` becomes an object hierarchy, this factory would
    make dimension class choices as well.
    """

    def __init__(self, dimension_dicts):
        self._dimension_dicts = dimension_dicts

    @classmethod
    def iter_dimensions(cls, dimension_dicts):
        """Generate Dimension object for each of *dimension_dicts*."""
        return cls(dimension_dicts)._iter_dimensions()

    def _iter_dimensions(self):
        """Generate Dimension object for each dimension dict."""
        return (
            Dimension(raw_dimension.dimension_dict, raw_dimension.dimension_type)
            for raw_dimension in self._raw_dimensions
        )

    @lazyproperty
    def _raw_dimensions(self):
        """Sequence of _RawDimension objects wrapping each dimension dict."""
        return tuple(
            _RawDimension(dimension_dict, self._dimension_dicts)
            for dimension_dict in self._dimension_dicts
        )


class _RawDimension(object):
    """Thin wrapper around dimension-dict to support dimension-type discovery.

    Determining dimension-type is pretty complex and requires repeated
    partial parsing of both the dimension dict and its siblings. This class
    abstracts that access for clarity.
    """

    def __init__(self, dimension_dict, dimension_dicts):
        self._dimension_dict = dimension_dict
        self._dimension_dicts = dimension_dicts

    @lazyproperty
    def dimension_dict(self):
        """dict defining this dimension in cube response."""
        return self._dimension_dict

    @lazyproperty
    def dimension_type(self):
        """Return member of DIMENSION_TYPE appropriate to dimension_dict."""
        base_type = self._base_type
        if base_type == "categorical":
            return self._resolve_categorical()
        if base_type == "enum.variable":
            return self._resolve_array_type()
        if base_type == "enum.datetime":
            return DT.DATETIME
        if base_type == "enum.numeric":
            return DT.BINNED_NUMERIC
        if base_type == "enum.text":
            return DT.TEXT
        raise NotImplementedError("unrecognized dimension type %s" % base_type)

    @lazyproperty
    def _alias(self):
        """Return str key for variable behind *dimension_dict*."""
        return self._dimension_dict["references"]["alias"]

    @lazyproperty
    def _base_type(self):
        """Return str like 'enum.numeric' representing dimension type.

        This string is a 'type.subclass' concatenation of the str keys
        used to identify the dimension type in the cube response JSON.
        The '.subclass' suffix only appears where a subtype is present.
        """
        type_class = self._dimension_dict["type"]["class"]
        if type_class == "categorical":
            return "categorical"
        if type_class == "enum":
            subclass = self._dimension_dict["type"]["subtype"]["class"]
            return "enum.%s" % subclass
        raise NotImplementedError("unexpected dimension type class '%s'" % type_class)

    @lazyproperty
    def _has_selected_category(self):
        """True if dimension-dict includes one or more selected categories.

        A "selected" category-dict is one having `'selected': True`. This
        property is only meaningful for a categorical dimension dict.
        """
        return True in {
            category.get("selected")
            for category in self._dimension_dict["type"].get("categories", [])
        }

    @lazyproperty
    def _is_array_cat(self):
        """True if a categorical dimension_dict belongs to an array pair.

        Returns True for a CA_CAT or MR_CAT dimension. Only meaningful when
        the dimension is known to be categorical (has base-type
        'categorical').
        """
        return "subreferences" in self._dimension_dict["references"]

    @lazyproperty
    def _next_raw_dimension(self):
        """_RawDimension for next *dimension_dict* in sequence or None for last.

        Returns None if this dimension is the last in sequence for this cube.
        """
        dimension_dicts = self._dimension_dicts
        this_idx = dimension_dicts.index(self._dimension_dict)
        if this_idx > len(dimension_dicts) - 2:
            return None
        return _RawDimension(dimension_dicts[this_idx + 1], self._dimension_dicts)

    def _resolve_array_type(self):
        """Return one of the ARRAY_TYPES members of DIMENSION_TYPE.

        This method distinguishes between CA and MR dimensions. The return
        value is only meaningful if the dimension is known to be of array
        type (i.e. either CA or MR, base-type 'enum.variable').
        """
        next_raw_dimension = self._next_raw_dimension
        if next_raw_dimension is None:
            return DT.CA

        is_mr_subvar = (
            next_raw_dimension._base_type == "categorical"
            and next_raw_dimension._has_selected_category
            and next_raw_dimension._alias == self._alias
        )
        return DT.MR if is_mr_subvar else DT.CA

    def _resolve_categorical(self):
        """Return one of the categorical members of DIMENSION_TYPE.

        This method distinguishes between CAT, CA_CAT, MR_CAT, and LOGICAL
        dimension types, all of which have the base type 'categorical'. The
        return value is only meaningful if the dimension is known to be one
        of the categorical types (has base-type 'categorical').
        """
        # ---an array categorical is either CA_CAT or MR_CAT---
        if self._is_array_cat:
            return DT.MR_CAT if self._has_selected_category else DT.CA_CAT

        # ---what's left is logical or plain-old categorical---
        return DT.LOGICAL if self._has_selected_category else DT.CAT


class Dimension(object):
    """Represents one dimension of a cube response.

    Each dimension represents one of the variables in a cube response. For
    example, a query to cross-tabulate snack-food preference against region
    will have two variables (snack-food preference and region) and will produce
    a two-dimensional (2D) cube response. That cube will have two of these
    dimension objects, which are accessed using
    :attr:`.CrunchCube.dimensions`.
    """

    def __init__(self, dimension_dict, dimension_type):
        self._dimension_dict = dimension_dict
        self._dimension_type = dimension_type

    @lazyproperty
    def all_elements(self):
        """_AllElements object providing cats or subvars of this dimension."""
        return _AllElements(self._dimension_dict["type"])

    @lazyproperty
    def description(self):
        """str description of this dimension."""
        description = self._dimension_dict["references"].get("description")
        return description if description else ""

    @lazyproperty
    def dimension_type(self):
        """Member of DIMENSION_TYPE appropriate to this cube dimension."""
        return self._dimension_type

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
        if self.dimension_type in {DT.MR_CAT, DT.LOGICAL}:
            return ()

        return tuple(
            (subtotal.anchor_idx, subtotal.addend_idxs) for subtotal in self._subtotals
        )

    @lazyproperty
    def inserted_hs_indices(self):
        """list of int index of each inserted subtotal for the dimension.

        Each value represents the position of a subtotal in the interleaved
        sequence of elements and subtotals items.
        """
        # ---don't do H&S insertions for CA and MR subvar dimensions---
        if self.dimension_type in DT.ARRAY_TYPES:
            return []

        return [
            idx
            for idx, item in enumerate(
                self._iter_interleaved_items(self.valid_elements)
            )
            if item.is_insertion
        ]

    @lazyproperty
    def is_marginable(self):
        """True if adding counts across this dimension axis is meaningful."""
        return self.dimension_type not in {DT.CA, DT.MR, DT.MR_CAT, DT.LOGICAL}

    def labels(
        self, include_missing=False, include_transforms=False, include_cat_ids=False
    ):
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
        elements = self.all_elements if include_missing else self.valid_elements

        include_subtotals = include_transforms and self.dimension_type != DT.CA_SUBVAR

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
        refs = self._dimension_dict["references"]
        return refs.get("name", refs.get("alias"))

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
        return tuple(element.numeric_value for element in self.valid_elements)

    @lazyproperty
    def shape(self):
        return len(self.all_elements)

    @lazyproperty
    def valid_elements(self):
        """_Elements object providing access to non-missing elements.

        Any categories or subvariables representing missing data are excluded
        from the collection; this sequence represents a subset of that
        provided by `.all_elements`.
        """
        return self.all_elements.valid_elements

    def _iter_interleaved_items(self, elements):
        """Generate element or subtotal items in interleaved order.

        This ordering corresponds to how value "rows" (or columns) are to
        appear after subtotals have been inserted at their anchor locations.
        Where more than one subtotal is anchored to the same location, they
        appear in their document order in the cube response.

        Only elements in the passed *elements* collection appear, which
        allows control over whether missing elements are included by choosing
        `.all_elements` or `.valid_elements`.
        """
        subtotals = self._subtotals

        for subtotal in subtotals.iter_for_anchor("top"):
            yield subtotal

        for element in elements:
            yield element
            for subtotal in subtotals.iter_for_anchor(element.element_id):
                yield subtotal

        for subtotal in subtotals.iter_for_anchor("bottom"):
            yield subtotal

    @lazyproperty
    def _subtotals(self):
        """_Subtotals sequence object for this dimension.

        The subtotals sequence provides access to any subtotal insertions
        defined on this dimension.
        """
        view = self._dimension_dict.get("references", {}).get("view", {})
        # ---view can be both None and {}, thus the edge case.---
        insertion_dicts = (
            [] if view is None else view.get("transform", {}).get("insertions", [])
        )
        return _Subtotals(insertion_dicts, self.valid_elements)


class NewDimension(Dimension):
    """A new version of Dimension that knows about its analysis-level transforms.

    The original Dimension object knew about its *base*-transforms. This new version
    adds the cascading behavior of analysis-specific transforms that elaborate those
    appearing on the cube-response itself.

    The idea is that soon we can just merge the properties of this class into the main
    Dimension class and get rid of this one. That will have to wait until FrozenSlice
    loads its own data and dimensions from the cube response rather than relying on
    CrunchCube to do that.
    """

    def __init__(self, dimension, dimension_transforms_dict):
        super(NewDimension, self).__init__(
            dimension._dimension_dict, dimension._dimension_type
        )
        self._dimension_transforms_dict = dimension_transforms_dict

    @lazyproperty
    def all_elements(self):
        """_NewAllElements object providing cats or subvars of this dimension.

        Elements in this sequence appear in cube-result order. Display order (including
        resolution of the explicit-reordering transforms cascade) is provided by
        a separate `.display_order` attribute on _NewAllElements.
        """
        return _NewAllElements(
            self._dimension_dict["type"], self._dimension_transforms_dict
        )

    @lazyproperty
    def description(self):
        """str description of this dimension."""
        # ---First authority in cascade is analysis-specific dimension transform. None
        # ---is a legitimate value, indicating suppression of any inherited subtitle.
        if "description" in self._dimension_transforms_dict:
            description = self._dimension_transforms_dict["description"]
        # ---inherited value is base dimension description---
        else:
            description = self._dimension_dict["references"].get("description")

        # ---Normalize to "" so return value is always a str and callers don't need to
        # ---deal with None as a possible return type.
        return description if description else ""

    @lazyproperty
    def display_order(self):
        """Sequence of int element-idx in order elements should be displayed.

        The sequence is exhaustive, containing an element index for each element in the
        `.all_elements` collection. The sequence reflects the resolved cascade of
        *explicit* ordering transforms, but does *not* reflect any *sort* transforms,
        which cannot be resolved by the dimension. Use the `.sort` property to access
        any sort transform that may apply.
        """
        return self.all_elements.display_order

    # TODO: Properly extract
    @lazyproperty
    def valid_display_order(self):
        return self.all_elements.valid_display_order

    @lazyproperty
    def name(self):
        """str name of this dimension, the empty string ("") if not specified."""
        references = self._dimension_dict["references"]
        # ---First authority in cascade is analysis-specific dimension transform. None
        # ---is a legitimate value, indicating suppression of any inherited title.
        if "name" in self._dimension_transforms_dict:
            name = self._dimension_transforms_dict["name"]
        # ---next authority is base dimension name---
        elif "name" in references:
            name = references["name"]
        else:
            name = references.get("alias")

        # ---Normalize None value to "" so return value is always a str and callers
        # ---don't need to deal with multiple possible return value types.
        return name if name else ""

    @lazyproperty
    def prune(self):
        """True if empty elements should be automatically hidden on this dimension."""
        prune = self._dimension_transforms_dict.get("prune")
        if prune is True:
            return True
        return False

    @lazyproperty
    def sort(self):
        """A _BaseSort-subclass object or None, describing the applied sort method.

        This value is None if no sort transform was specified for this dimension.
        Currently that is its only possible value. The returned sort object describes
        the sort method which can include sorting on the value of an opposing element or
        on the margin and specify ascending or descending order.
        """
        return None

    @lazyproperty
    def subtotals(self):
        """_Subtotals sequence object for this dimension.

        Each item in the sequence is a _Subtotal object specifying a subtotal, including
        its addends and anchor.
        """
        # ---insertions present in the dimension-transforms override those on the
        # ---dimension itself.
        insertion_dicts = self._dimension_transforms_dict.get("insertions")
        if insertion_dicts is not None:
            return _Subtotals(insertion_dicts, self.valid_elements)

        # ---otherwise, insertions defined as default transforms apply---
        view = self._dimension_dict.get("references", {}).get("view", {})
        # ---view can be both None and {}, thus the edge case.---
        insertion_dicts = (
            [] if view is None else view.get("transform", {}).get("insertions", [])
        )
        return _Subtotals(insertion_dicts, self.valid_elements)

    # NOTE: The Assembler object should be the only one doing any interleaving. Remove
    # this method after FrozenSlice integration.
    def _iter_interleaved_items(self, elements):
        raise NotImplementedError("you don't need this")


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
        raise NotImplementedError("must be implemented by each subclass")

    @lazyproperty
    def _element_makings(self):
        """(ElementCls, element_dicts) pair for this dimension's elements.

        All the elements of a given dimension are the same type. This method
        determines the type (class) and source dicts for the elements of this
        dimension and provides them for the element factory.
        """
        if self._type_dict["class"] == "categorical":
            return _Category, self._type_dict["categories"]
        return _Element, self._type_dict["elements"]

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
            ElementCls(element_dict, idx, element_dicts)
            for idx, element_dict in enumerate(element_dicts)
        )


class _NewAllElements(_AllElements):
    # TODO: add specialized constructor, perhaps reusing existing element factory or
    # something. Pass element-transforms-dict down to each _NewElement thing.

    def __init__(self, type_dict, dimension_transforms_dict):
        self._type_dict = type_dict
        self._dimension_transforms_dict = dimension_transforms_dict

    @lazyproperty
    def display_order(self):
        """Sequence of int element-idx reflecting order in which to display elements.

        This order reflects the application of any explicit element-order transforms,
        including resolution of any cascade. It does *not* reflect the results of
        a *sort* transform, which can only be resolved at a higher level, where vector
        values are known.
        """
        explicit_order = self._explicit_order
        if explicit_order:
            return explicit_order
        return tuple(range(len(self._element_dicts)))

    # TODO: Need to properly extract this, probably to `NewValidElements`
    @lazyproperty
    def valid_display_order(self):
        explicit_order = self._valid_explicit_order
        if explicit_order:
            return explicit_order
        return tuple(range(len(self.valid_elements)))

    @lazyproperty
    def _element_dicts(self):
        """Sequence of element-dicts for this dimension, taken from cube-result."""
        return (
            self._type_dict["categories"]
            if self._type_dict["class"] == "categorical"
            else self._type_dict["elements"]
        )

    @lazyproperty
    def _elements(self):
        """Composed tuple storing actual sequence of element objects."""
        element_dicts = self._element_dicts
        return tuple(
            _NewElement(element_dict, idx, element_dicts, element_transforms_dict)
            for (
                idx,
                element_dict,
                element_transforms_dict,
            ) in self._iter_element_makings()
        )

    # TODO: extract this to a function object. It's too complex the way it is, doing
    # like seven separate things.
    @lazyproperty
    def _explicit_order(self):
        """Sequence of int element idx or None, reflecting explicit-order transform.

        This value is None if no explicit-order transform is specified. Otherwise, it is
        an exhaustive collection of element offsets, in the order specified (and in some
        cases implied) by the order transform.
        """
        order_dict = self._dimension_transforms_dict.get("order", {})
        order_type = order_dict.get("type")
        if order_type != "explicit":
            return None
        ordered_element_ids = order_dict.get("element_ids")
        if not isinstance(ordered_element_ids, list):
            return None

        # ---list like [0, 1, 2, -1], perhaps ["0001", "0002", etc.]---
        cube_result_order = list(element.element_id for element in self)
        remaining_element_ids = list(element.element_id for element in self)

        # print("default_order == %s" % default_order)

        ordered_idxs = []
        for element_id in ordered_element_ids:
            # ---An element-id appearing in transform but not in dimension is ignored.
            # ---Also, a duplicated element-id is only used on first encounter.
            if element_id not in remaining_element_ids:
                continue
            ordered_idxs.append(cube_result_order.index(element_id))
            remaining_element_ids.remove(element_id)

        # ---any remaining element-ids are tacked onto the end of the list in the order
        # ---they originally appeared in the cube-result.
        for element_id in remaining_element_ids:
            ordered_idxs.append(cube_result_order.index(element_id))

        return tuple(ordered_idxs)

    # TODO: We should only need this. I've just copy/pasted it. It needs to be
    # wrapped up nicely
    @lazyproperty
    def _valid_explicit_order(self):
        """Sequence of int element idx or None, reflecting explicit-order transform.

        This value is None if no explicit-order transform is specified. Otherwise, it is
        an exhaustive collection of element offsets, in the order specified (and in some
        cases implied) by the order transform.
        """
        order_dict = self._dimension_transforms_dict.get("order", {})
        order_type = order_dict.get("type")
        if order_type != "explicit":
            return None
        ordered_element_ids = order_dict.get("element_ids")
        if not isinstance(ordered_element_ids, list):
            return None

        # ---list like [0, 1, 2, -1], perhaps ["0001", "0002", etc.]---
        cube_result_order = list(
            element.element_id for element in self if not element.missing
        )
        remaining_element_ids = list(
            element.element_id for element in self if not element.missing
        )

        # print("default_order == %s" % default_order)

        ordered_idxs = []
        for element_id in ordered_element_ids:
            if element_id not in remaining_element_ids:
                continue
            ordered_idxs.append(cube_result_order.index(element_id))
            remaining_element_ids.remove(element_id)

        for element_id in remaining_element_ids:
            ordered_idxs.append(cube_result_order.index(element_id))

        return tuple(ordered_idxs)

    def _iter_element_makings(self):
        """Generate tuple of values needed to construct each element object.

        An (idx, element_dict, element_transforms_dict) tuple is generated for each
        element in this dimension, in the order they appear in the cube-result. All
        elements are included (including missing).
        """
        element_dicts = self._element_dicts
        elements_transforms = self._dimension_transforms_dict.get("elements", {})
        for idx, element_dict in enumerate(element_dicts):
            element_id = element_dict["id"]
            element_transforms_dict = elements_transforms.get(element_id, {})
            yield idx, element_dict, element_transforms_dict


class _ValidElements(_BaseElements):
    """Sequence of non-missing element objects for a dimension.

    *all_elements* is an instance of _AllElements containing all the elements
    of a dimension. This object is only intended to be constructed by
    _AllElements.valid_elements and there should be no reason to construct it
    directly.
    """

    def __init__(self, all_elements):
        self.all_elements = all_elements

    @lazyproperty
    def _elements(self):
        """tuple containing actual sequence of element objects."""
        return tuple(element for element in self.all_elements if not element.missing)


class _BaseElement(object):
    """Base class for element objects."""

    def __init__(self, element_dict, index, element_dicts):
        self._element_dict = element_dict
        self._index = index
        self._element_dicts = element_dicts

    @lazyproperty
    def element_id(self):
        """int identifier for this category or subvariable."""
        return self._element_dict["id"]

    @lazyproperty
    def index(self):
        """int offset at which this element appears in dimension.

        This position is based upon the document position of this element in
        the cube response. No adjustment for missing elements is made.
        """
        return self._index

    @lazyproperty
    def index_in_valids(self):
        valid_ids = [el["id"] for el in self._element_dicts if not el.get("missing")]
        return valid_ids.index(self.element_id)

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
        return bool(self._element_dict.get("missing"))

    @lazyproperty
    def numeric_value(self):
        """Numeric value assigned to element by user, np.nan if absent."""
        numeric_value = self._element_dict.get("numeric_value")
        return np.nan if numeric_value is None else numeric_value


class _NewElement(_BaseElement):
    """A new version of _BaseElement that knows about its analysis-level transforms.

    This will become _Element after integration and will replace all of _BaseElement,
    _Category, and _Element. The distinction between category and element is only in the
    schema location for the label, and that's not enough difference to justify
    subclassing.

    This new version adds the cascading behavior of analysis-specific element transforms
    like name and hide.

    The idea is that soon we can just merge the properties of this class into the main
    Dimension class and get rid of this one. That will have to wait until FrozenSlice
    loads its own data and dimensions from the cube response rather than relying on
    CrunchCube to do that.
    """

    def __init__(self, element_dict, index, element_dicts, element_transforms_dict):
        super(_NewElement, self).__init__(element_dict, index, element_dicts)
        self._element_transforms_dict = element_transforms_dict

    @lazyproperty
    def is_hidden(self):
        """True if this element is explicitly hidden in this analysis."""
        # ---first authority is hide transform in element transforms---
        if "hide" in self._element_transforms_dict:
            return self._element_transforms_dict["hide"] is True
        # ---default is not hidden, there is currently no prior-level hide transform---
        return False

    # NOTE: I'm thinking this one goes away in the FrozenSlice world
    @lazyproperty
    def is_insertion(self):
        raise NotImplementedError

    @lazyproperty
    def label(self):
        """str display-name for this element.

        This value is the empty string when no value has been specified or display of
        the name has been suppressed.

        This property handles elements for variables of all types, including
        categorical, array (subvariable), numeric, datetime and text.
        """
        # ---first authority is name transform in element transforms---
        if "name" in self._element_transforms_dict:
            name = self._element_transforms_dict["name"]
            return name if name else ""

        # ---otherwise base-name from element-dict is used---
        element_dict = self._element_dict

        # ---category elements have a name item---
        if "name" in element_dict:
            name = element_dict["name"]
            return name if name else ""

        # ---other types are more complicated---
        value = element_dict.get("value")
        type_name = type(value).__name__

        if type_name == "NoneType":
            return ""

        if type_name == "list":
            # ---like '10-15' or 'A-F'---
            return "-".join([str(item) for item in value])

        if type_name in ("float", "int"):
            return str(value)

        if type_name in ("str", "unicode"):
            return value

        # ---For CA and MR subvar dimensions---
        name = value.get("references", {}).get("name")
        return name if name else ""


class _Category(_BaseElement):
    """A category on a categorical dimension."""

    def __init__(self, category_dict, index, element_dicts):
        super(_Category, self).__init__(category_dict, index, element_dicts)
        self._category_dict = category_dict

    @lazyproperty
    def label(self):
        """str display name assigned to this category by user."""
        name = self._category_dict.get("name")
        return name if name else ""


class _Element(_BaseElement):
    """A subvariable on an MR or CA enum dimension."""

    @lazyproperty
    def label(self):
        """str display-name for this element, '' when absent from cube response.

        This property handles numeric, datetime and text variables, but also
        subvar dimensions
        """
        value = self._element_dict.get("value")
        type_name = type(value).__name__

        if type_name == "NoneType":
            return ""

        if type_name == "list":
            # ---like '10-15' or 'A-F'---
            return "-".join([str(item) for item in value])

        if type_name in ("float", "int"):
            return str(value)

        if type_name in ("str", "unicode"):
            return value

        # ---For CA and MR subvar dimensions---
        name = value.get("references", {}).get("name")
        return name if name else ""


class _Subtotals(Sequence):
    """Sequence of _Subtotal objects for a dimension.

    Each _Subtotal object represents a "subtotal" insertion transformation
    defined for the dimension.

    A subtotal can only involve valid (i.e. non-missing) elements.
    """

    def __init__(self, insertion_dicts, valid_elements):
        self._insertion_dicts = insertion_dicts
        self.valid_elements = valid_elements

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
        return (subtotal for subtotal in self._subtotals if subtotal.anchor == anchor)

    @lazyproperty
    def _element_ids(self):
        """frozenset of int id of each non-missing cat or subvar in dim."""
        return frozenset(self.valid_elements.element_ids)

    def _iter_valid_subtotal_dicts(self):
        """Generate each insertion dict that represents a valid subtotal."""
        for insertion_dict in self._insertion_dicts:
            # ---skip any non-dicts---
            if not isinstance(insertion_dict, dict):
                continue

            # ---skip any non-subtotal insertions---
            if insertion_dict.get("function") != "subtotal":
                continue

            # ---skip any malformed subtotal-dicts---
            if not {"anchor", "args", "name"}.issubset(insertion_dict.keys()):
                continue

            # ---skip if doesn't reference at least one non-missing element---
            if not self._element_ids.intersection(insertion_dict["args"]):
                continue

            # ---an insertion-dict that successfully runs this gauntlet
            # ---is a valid subtotal dict
            yield insertion_dict

    @lazyproperty
    def _subtotals(self):
        """Composed tuple storing actual sequence of _Subtotal objects."""
        return tuple(
            _Subtotal(subtotal_dict, self.valid_elements)
            for subtotal_dict in self._iter_valid_subtotal_dicts()
        )


class _Subtotal(object):
    """A subtotal insertion on a cube dimension."""

    def __init__(self, subtotal_dict, valid_elements):
        self._subtotal_dict = subtotal_dict
        self.valid_elements = valid_elements

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
        anchor = self._subtotal_dict["anchor"]

        if anchor is None:
            # In the case of undefined anchor default to "bottom"
            return "bottom"

        try:
            anchor = int(anchor)
            if anchor not in self.valid_elements.element_ids:
                # In the case of a non-valid int id, default to "bottom"
                return "bottom"
            return anchor
        except (TypeError, ValueError):
            return anchor.lower()

    @lazyproperty
    def anchor_idx(self):
        """int or str representing index of anchor element in dimension.

        When the anchor is an operation, like 'top' or 'bottom'
        """
        anchor = self.anchor
        if anchor in ["top", "bottom"]:
            return anchor
        return self.valid_elements.get_by_id(anchor).index_in_valids

    @lazyproperty
    def addend_ids(self):
        """tuple of int ids of elements contributing to this subtotal.

        Any element id not present in the dimension or present but
        representing missing data is excluded.
        """
        return tuple(
            arg
            for arg in self._subtotal_dict.get("args", [])
            if arg in self.valid_elements.element_ids
        )

    @lazyproperty
    def addend_idxs(self):
        """tuple of int index of each addend element for this subtotal.

        The length of the tuple is the same as that for `.addend_ids`, but
        each value repesents the offset of that element within the dimension,
        rather than its element id.
        """
        return tuple(
            self.valid_elements.get_by_id(addend_id).index_in_valids
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
        name = self._subtotal_dict.get("name")
        return name if name else ""
