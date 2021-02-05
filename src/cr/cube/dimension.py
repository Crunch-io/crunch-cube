# encoding: utf-8

"""Provides the Dimension class."""

import sys

if sys.version_info >= (3, 3):
    from collections.abc import Sequence  # pragma: no cover
else:
    from collections import Sequence  # pragma: no cover

import numpy as np

from cr.cube.enums import COLLATION_METHOD as CM, DIMENSION_TYPE as DT
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
        raise NotImplementedError(
            "must be implemented by each subclass"
        )  # pragma: no cover


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
        if base_type == "enum.num_arr":
            return DT.NUM_ARRAY
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

    def __init__(self, dimension_dict, dimension_type, dimension_transforms=None):
        self._dimension_dict = dimension_dict
        self._dimension_type = dimension_type
        self._dimension_transforms_dict = dimension_transforms or {}

    @lazyproperty
    def alias(self):
        """Return the alias for the dimension if it exists, None otherwise."""
        return self._dimension_dict["references"].get("alias")

    @lazyproperty
    def all_elements(self):
        """_AllElements object providing cats or subvars of this dimension.

        Elements in this sequence appear in cube-result order. Display order (including
        resolution of the explicit-reordering transforms cascade) is provided by
        a separate `.display_order` attribute on _AllElements.
        """
        return _AllElements(
            self._dimension_dict["type"], self._dimension_transforms_dict
        )

    def apply_transforms(self, dimension_transforms):
        """Return a new `Dimension` object with `dimension_transforms` applied.

        The new dimension object is the same as this one in all other respects.
        """
        return Dimension(
            self._dimension_dict, self._dimension_type, dimension_transforms
        )

    @lazyproperty
    def collation_method(self):
        """Member of COLLATION_METHOD specifying ordering of dimension elements."""
        method_keyword = self.order_dict.get("type")
        if method_keyword is None:
            return CM.PAYLOAD_ORDER
        return CM(method_keyword)

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
    def dimension_type(self):
        """Member of DIMENSION_TYPE appropriate to this cube dimension."""
        return self._dimension_type

    @lazyproperty
    def display_order(self):
        """Sequence of int element indices specifying display order of elements.

        The sequence includes only valid elements; missing elements do not appear.
        Further, each index represents the document-order position of the element in the
        sequence of valid elements; missing elements are skipped in the assignment of
        indexes. The returned sequence is exhaustive; all valid elements are
        represented.

        The sequence reflects the resolved cascade of any *explicit* ordering
        transforms, but does *not* reflect any *sort* transforms, which cannot be
        resolved by the dimension. Use the `.sort` property to access any sort transform
        that may apply.

        Example with explicit-order transform:

            (3, 0, 2, 1, 4)

        Example with no explicit-order transform:

            (0, 1, 2, 3, 4)
        """
        return self.valid_elements.display_order

    @lazyproperty
    def element_ids(self):
        """tuple of int element-id for each valid element in this dimension.

        Element-ids appear in the order defined in the cube-result.
        """
        return tuple(e.element_id for e in self.valid_elements)

    @lazyproperty
    def hidden_idxs(self):
        """tuple of int element-idx for each hidden valid element in this dimension.

        An element is hidden when a "hide" transform is applied to it in its transforms
        dict.
        """
        return tuple(
            idx for idx, element in enumerate(self.valid_elements) if element.is_hidden
        )

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
    def order_dict(self):
        """dict dimension.transforms.order field parsed from JSON payload.

        Value is `{}` if no "order": field is present.
        """
        return self._dimension_transforms_dict.get("order", {})

    @lazyproperty
    def prune(self):
        """True if empty elements should be automatically hidden on this dimension."""
        prune = self._dimension_transforms_dict.get("prune")
        if prune is True:
            return True
        return False

    @lazyproperty
    def selected_categories(self):
        """List of selected categories specified for this dimension."""
        selected_categories = self._dimension_dict["references"].get(
            "selected_categories"
        )
        return tuple(selected_categories) if selected_categories else ()

    @lazyproperty
    def shape(self):
        """int count of *all* elements in this dimension, both valid and missing."""
        return len(self.all_elements)

    @lazyproperty
    def subtotals(self):
        """_Subtotals sequence object for this dimension.

        Each item in the sequence is a _Subtotal object specifying a subtotal, including
        its addends and anchor.
        """
        # --- elements of an aggregate/array dimension cannot meaningfully be summed, so
        # --- an array dimension cannot have subtotals
        if self.dimension_type in (DT.MR, DT.CA_SUBVAR):
            insertion_dicts = []
        # --- insertions in dimension-transforms override those on dimension itself ---
        elif "insertions" in self._dimension_transforms_dict:
            insertion_dicts = self._dimension_transforms_dict["insertions"]
        # --- otherwise insertions defined on dimension/variable apply ---
        else:
            view = self._dimension_dict.get("references", {}).get("view") or {}
            insertion_dicts = view.get("transform", {}).get("insertions", [])

        return _Subtotals(insertion_dicts, self.valid_elements)

    @lazyproperty
    def valid_elements(self):
        """_Elements object providing access to non-missing elements.

        Any categories or subvariables representing missing data are excluded
        from the collection; this sequence represents a subset of that
        provided by `.all_elements`.
        """
        return self.all_elements.valid_elements


class _BaseElements(Sequence):
    """Base class for element sequence containers."""

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
    def _elements_by_id(self):
        """dict mapping each element by its id."""
        return {element.element_id: element for element in self._elements}


class _AllElements(_BaseElements):
    """Sequence of _BaseElement subclass objects for a dimension.

    Each element is either a category or a subvariable.
    """

    def __init__(self, type_dict, dimension_transforms_dict):
        self._type_dict = type_dict
        self._dimension_transforms_dict = dimension_transforms_dict

    @lazyproperty
    def valid_elements(self):
        """_ValidElements object containing only non-missing elements."""
        return _ValidElements(self._elements, self._dimension_transforms_dict)

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
        """tuple storing actual sequence of element objects."""
        return tuple(
            _Element(
                element_dict,
                idx,
                _ElementTransforms(element_transforms_dict, self._prune),
            )
            for (
                idx,
                element_dict,
                element_transforms_dict,
            ) in self._iter_element_makings()
        )

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
            # TODO: Each element transforms dict is keyed by the str() version of it int
            # value as a consequence of JSON serialization (which does not allow
            # non-string keys). Hence the str(element_id) here.
            element_transforms_dict = elements_transforms.get(
                element_id, elements_transforms.get(str(element_id), {})
            )
            yield idx, element_dict, element_transforms_dict

    @lazyproperty
    def _prune(self):
        """True if empty elements in this dimension should be automatically hidden."""
        return True if self._dimension_transforms_dict.get("prune") is True else False


class _ValidElements(_BaseElements):
    """Sequence of non-missing element objects for a dimension.

    *all_elements* is an instance of _AllElements containing all the elements
    of a dimension. This object is only intended to be constructed by
    _AllElements.valid_elements and there should be no reason to construct it
    directly.
    """

    def __init__(self, all_elements, dimension_transforms_dict):
        self._all_elements = all_elements
        self._dimension_transforms_dict = dimension_transforms_dict

    @lazyproperty
    def display_order(self):
        """Sequence of int element-idx reflecting order in which to display elements.

        This order reflects the application of any explicit element-order transforms,
        including resolution of any cascade. It does *not* reflect the results of
        a *sort* transform, which can only be resolved at a higher level, where vector
        values are known.
        """
        return (
            self._explicit_order
            if self._explicit_order
            else tuple(range(len(self._elements)))
        )

    @lazyproperty
    def _elements(self):
        """tuple containing actual sequence of element objects."""
        return tuple(element for element in self._all_elements if not element.missing)

    @lazyproperty
    def _explicit_order(self):
        """Sequence of int element-idx or None, reflecting explicit-order transform.

        This value is None if no explicit-order transform is specified. Otherwise, it is
        an exhaustive collection of (valid) element offsets, in the order specified (and
        in some cases implied) by the order transform.
        """
        # ---get order transform if any, aborting if no explicit order transform---
        order_dict = self._dimension_transforms_dict.get("order", {})
        order_type = order_dict.get("type")
        ordered_element_ids = order_dict.get("element_ids")
        if order_type != "explicit" or not isinstance(ordered_element_ids, list):
            return None

        # ---list like [0, 1, 2, -1], perhaps ["0001", "0002", etc.], reflecting element
        # ---ids in the order they appear in the cube result. We'll use this to map
        # ---element-id to its index in the valid-elements sequence.
        cube_result_order = tuple(element.element_id for element in self)
        # ---this is a copy of the same, but we're going to mutate this one. This is
        # ---required to implement the "no-duplicates" behavior.
        remaining_element_ids = list(cube_result_order)

        # ---we'll collect the results in this---
        ordered_idxs = []
        # ---append idx of each element mentioned by id in transform, in order. Remove
        # ---each element-id from remaining as we go to keep track of dups and leftovers
        for element_id in ordered_element_ids:
            # ---An element-id appearing in transform but not in dimension is ignored.
            # ---Also, a duplicated element-id is only used on first encounter.
            if element_id not in remaining_element_ids:
                continue
            ordered_idxs.append(cube_result_order.index(element_id))
            remaining_element_ids.remove(element_id)

        # ---any remaining elements are tacked onto the end of the list in the order
        # ---they originally appeared in the cube-result.
        for element_id in remaining_element_ids:
            ordered_idxs.append(cube_result_order.index(element_id))

        return tuple(ordered_idxs)


class _Element(object):
    """A category or subvariable of a dimension.

    This object resolves the transform cascade for element-level transforms.
    """

    def __init__(self, element_dict, index, element_transforms):
        self._element_dict = element_dict
        self._index = index
        self._element_transforms = element_transforms

    @lazyproperty
    def element_id(self):
        """int identifier for this category or subvariable."""
        return self._element_dict["id"]

    @lazyproperty
    def fill(self):
        """str RGB color like "#af032d" or None if not specified.

        A value of None indicates the default fill should be used for this element.
        A str value must be a hash character ("#") followed by six hexadecimal digits.
        Three-character color contractions (like "#D07") are not valid.
        """
        # ---first authority is fill transform in element transforms. The base value is
        # ---the only prior authority and that is None (use default fill color).
        return self._element_transforms.fill

    @lazyproperty
    def index(self):
        """int offset at which this element appears in dimension.

        This position is based upon the document position of this element in
        the cube response. No adjustment for missing elements is made.
        """
        return self._index

    @lazyproperty
    def is_hidden(self):
        """True if this element is explicitly hidden in this analysis."""
        # ---first authority is hide transform in element transforms---
        # ---default is not hidden, there is currently no prior-level hide transform---
        return {True: True, False: False, None: False}[self._element_transforms.hide]

    @lazyproperty
    def label(self):
        """str display-name for this element.

        This value is the empty string when no value has been specified or display of
        the name has been suppressed.

        This property handles elements for variables of all types, including
        categorical, array (subvariable), numeric, datetime and text.
        """
        # ---first authority is name transform in element transforms---
        name = self._element_transforms.name
        if name is not None:
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

    @lazyproperty
    def prune(self):
        """True if this element should be hidden when empty, False otherwise."""
        return self._element_transforms.prune


class _ElementTransforms(object):
    """A value object providing convenient access to transforms for a single element."""

    def __init__(self, element_transforms_dict, prune):
        self._element_transforms_dict = element_transforms_dict
        self._prune = prune

    @lazyproperty
    def fill(self):
        """str RGB color like "#af032d" or None if not specified.

        A value of None indicates no fill transform was specified for this element.
        A str value must be a hash character ("#") followed by six hexadecimal digits.
        Three-character color contractions (like "#D07") are not valid.
        """

        fill = self._element_transforms_dict.get("fill")

        if not fill:
            return None
        return fill

    @lazyproperty
    def hide(self):
        """Tri-value, True if this element has been explicitly hidden in this analysis.

        False overrides any prior "hide" transform with "show" and None signifies
        "inherit".
        """
        hide = self._element_transforms_dict.get("hide")
        # ---cover all of "omitted", "==None", and odd "==[]" or "==''" cases---
        if hide is True:
            return True
        if hide is False:
            return False
        return None

    @lazyproperty
    def name(self):
        """str display-name for this element or None if not specified."""
        # ---if "name": element is omitted, no transform is specified---
        if "name" not in self._element_transforms_dict:
            return None
        # ---otherwise normalize value to str, with an explicit value of None, [], 0,
        # ---etc. becoming the empty string ("").
        name = self._element_transforms_dict["name"]
        return str(name) if name else ""

    @lazyproperty
    def prune(self):
        """True if this element should be hidden when empty."""
        return self._prune


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
            if insertion_dict.get("function") != "subtotal":
                continue

            # ---skip any hidden insertions---
            if insertion_dict.get("hide") is True:
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
            _Subtotal(subtotal_dict, self._valid_elements, idx + 1)
            for idx, subtotal_dict in enumerate(self._iter_valid_subtotal_dicts())
        )


class _Subtotal(object):
    """A subtotal insertion on a cube dimension.

    `fallback_insertion_id` is a fallback unique identifier for this insertion, until
    real insertion-ids can be added. Its value is just the index+1 of this subtotal
    within the insertions transform collection.
    """

    def __init__(self, subtotal_dict, valid_elements, fallback_insertion_id):
        self._subtotal_dict = subtotal_dict
        self._valid_elements = valid_elements
        self._fallback_insertion_id = fallback_insertion_id

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
            if anchor not in self._valid_elements.element_ids:
                # In the case of a non-valid int id, default to "bottom"
                return "bottom"
            return anchor
        except (TypeError, ValueError):
            return anchor.lower()

    @lazyproperty
    def addend_ids(self):
        """tuple of int ids of elements contributing to this subtotal.

        Any element id not present in the dimension or present but
        representing missing data is excluded.
        """
        return tuple(
            arg
            for arg in self._subtotal_dict.get("args", [])
            if arg in self._valid_elements.element_ids
        )

    @lazyproperty
    def addend_idxs(self):
        """ndarray of int base-element offsets contributing to this subtotal.

        Suitable for directly indexing a numpy array object (such as base values or
        margin) to extract the addend values for this subtotal.
        """
        addend_ids = self.addend_ids
        return np.fromiter(
            (
                idx
                for idx, vector in enumerate(self._valid_elements)
                if vector.element_id in addend_ids
            ),
            dtype=int,
        )

    @lazyproperty
    def insertion_id(self):
        """int unique identifier of this subtotal within this dimension's insertions."""
        return self._subtotal_dict.get("insertion_id", self._fallback_insertion_id)

    @lazyproperty
    def label(self):
        """str display name for this subtotal, suitable for use as label."""
        name = self._subtotal_dict.get("name")
        return name if name else ""
