# encoding: utf-8

"""Provides the Dimension class."""

import copy
from collections.abc import Sequence
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from cr.cube.enums import (
    COLLATION_METHOD as CM,
    DIMENSION_TYPE as DT,
    MARGINAL,
    MEASURE,
    _DimensionType,
)
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
    def _dimensions(self) -> Tuple["Dimension", ...]:
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
    def apparent_dimensions(self) -> "_ApparentDimensions":
        """_ApparentDimensions collection of the "visible" dimensions.

        The two dimensions for a multiple-response (MR) variable are
        conflated into a single dimensions in this collection.
        """
        return _ApparentDimensions(all_dimensions=self._dimensions)

    @lazyproperty
    def dimension_order(self) -> Tuple[int, ...]:
        """Tuple of int representing the dimension order.

        The dimension order depends on the presence of numeric array in the dimensions
        and the number of the cube dimensions. In case of 3 dimensions e.g.
        NUM_ARR_X_MR_SUBVAR_X_MR_CAT the order should be (1,2,0) that is basically
        swapping the MR (2 dimensions) with the NUM_ARRAY dimension. In case of 2
        dimensions the dimension order correspond simpy to the reverse of the original
        dimension order.
        """
        # NOTE: this is a temporary hack that goes away when we introduce the dim_order
        # concept. We should receive the actual order directly in the cube_response.
        # So, all this logic will be deleted.
        dimension_types = tuple(d.dimension_type for d in self._dimensions)
        dim_order = tuple(range(len(self._dimensions)))
        if len(self._dimensions) >= 2 and DT.NUM_ARRAY in dimension_types:
            return (
                dim_order[-2:] + (dim_order[0],)
                if len(self._dimensions) == 3
                else dim_order[::-1]
            )
        return dim_order

    @lazyproperty
    def shape(self) -> Tuple[int, ...]:
        """Tuple of int element count for each dimension.

        This corresponds to the shape of the ndarray representing the raw
        cube response values (raw meaning including missing and prunable
        elements and any MR_CAT dimensions).
        """
        dimensions = [self._dimensions[i] for i in self.dimension_order]
        return tuple(d.shape for d in dimensions)

    @lazyproperty
    def _dimensions(self) -> Tuple["Dimension", ...]:
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
    def _dimensions(self) -> Tuple["Dimension", ...]:
        """tuple of dimension objects in this collection.

        This composed tuple is the source for the dimension objects in this
        collection.
        """
        return tuple(d for d in self._all_dimensions if d.dimension_type != DT.MR_CAT)


class _DimensionFactory:
    """Produce Dimension objects of correct type from dimension-dicts.

    "type" here is primarily the `.dimension_type` value of the dimension,
    although if `Dimension` becomes an object hierarchy, this factory would
    make dimension class choices as well.
    """

    def __init__(self, dimension_dicts):
        self._dimension_dicts = dimension_dicts

    @classmethod
    def iter_dimensions(cls, dimension_dicts) -> Iterator["Dimension"]:
        """Generate Dimension object for each of *dimension_dicts*."""
        return cls(dimension_dicts)._iter_dimensions()

    def _iter_dimensions(self) -> Iterator["Dimension"]:
        """Generate Dimension object for each dimension dict."""
        return (
            Dimension(raw_dimension.dimension_dict, raw_dimension.dimension_type)
            for raw_dimension in self._raw_dimensions
        )

    @lazyproperty
    def _raw_dimensions(self) -> Tuple["_RawDimension", ...]:
        """Sequence of _RawDimension objects wrapping each dimension dict."""
        return tuple(
            _RawDimension(dimension_dict, self._dimension_dicts)
            for dimension_dict in self._dimension_dicts
        )


class _RawDimension:
    """Thin wrapper around dimension-dict to support dimension-type discovery.

    Determining dimension-type is pretty complex and requires repeated
    partial parsing of both the dimension dict and its siblings. This class
    abstracts that access for clarity.
    """

    def __init__(self, dimension_dict, dimension_dicts):
        self._dimension_dict = dimension_dict
        self._dimension_dicts = dimension_dicts

    @lazyproperty
    def dimension_dict(self) -> Dict:
        """dict defining this dimension in cube response."""
        return self._dimension_dict

    @lazyproperty
    def dimension_type(self) -> _DimensionType:
        """_DimensionType (member of DIMENSION_TYPE) appropriate to dimension_dict."""
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
        raise NotImplementedError(f"unrecognized dimension type {base_type}")

    @lazyproperty
    def _alias(self) -> str:
        """Return str key for variable behind *dimension_dict*."""
        return self._dimension_dict["references"]["alias"]

    @lazyproperty
    def _base_type(self) -> str:
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
            return f"enum.{subclass}"
        raise NotImplementedError(f"unexpected dimension type class '{type_class}'")

    @lazyproperty
    def _categories(self) -> List[Dict]:
        return self._dimension_dict["type"].get("categories", [])

    @lazyproperty
    def _has_selected_category(self) -> bool:
        """True if dimension-dict includes one or more selected categories.

        A "selected" category-dict is one having `'selected': True`. This
        property is only meaningful for a categorical dimension dict.
        """
        return any(category.get("selected") for category in self._categories)

    @lazyproperty
    def _is_logical_type(self) -> bool:
        """True if dimension-dict has the categories equal to those of the logical type.

        Logical type has exactly three categories with IDs [-1, 0, 1]. This type is
        used to define the selections dimension of the multiple response type, when it
        follows the subvariables dimension.
        """
        return [category.get("id") for category in self._categories] == [1, 0, -1]

    @lazyproperty
    def _is_array_cat(self) -> bool:
        """True if a categorical dimension_dict belongs to an array pair.

        Returns True for a CA_CAT or MR_CAT dimension. Only meaningful when
        the dimension is known to be categorical (has base-type
        'categorical').
        """
        return "subreferences" in self._dimension_dict["references"]

    @lazyproperty
    def _is_cat_date(self) -> bool:
        """True if dimension is a categorical date, False otherwise.

        A dimension is a categorical date, if it has all the properties of a "normal"
        categorical dimension, but also a `"date"` field in any of its categories.
        """
        if self._dimension_dict["type"]["class"] != "categorical":
            return False
        return any("date" in cat for cat in self._categories)

    @lazyproperty
    def _next_raw_dimension(self) -> Optional["_RawDimension"]:
        """_RawDimension for next *dimension_dict* in sequence or None for last.

        Returns None if this dimension is the last in sequence for this cube.
        """
        dimension_dicts = self._dimension_dicts
        this_idx = dimension_dicts.index(self._dimension_dict)
        if this_idx > len(dimension_dicts) - 2:
            return None
        return _RawDimension(dimension_dicts[this_idx + 1], self._dimension_dicts)

    def _resolve_array_type(self) -> _DimensionType:
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
            and next_raw_dimension._is_logical_type
        )
        return DT.MR if is_mr_subvar else DT.CA

    def _resolve_categorical(self) -> _DimensionType:
        """Return one of the categorical members of DIMENSION_TYPE.

        This method distinguishes between CAT, CA_CAT, MR_CAT, and LOGICAL
        dimension types, all of which have the base type 'categorical'. The
        return value is only meaningful if the dimension is known to be one
        of the categorical types (has base-type 'categorical').
        """
        # ---an array categorical is either CA_CAT or MR_CAT---
        if self._is_array_cat:
            return (
                DT.MR_CAT
                if self._has_selected_category and self._is_logical_type
                else DT.CA_CAT
            )

        # ---what's left is three different versions of categorical dimension---

        # ---first the logical---
        if self._has_selected_category and self._is_logical_type:
            return DT.LOGICAL

        # ---or a categorical date---
        if self._is_cat_date:
            return DT.CAT_DATE

        # ---or the plain-old categorical---
        return DT.CAT


class Dimension:
    """Represents one dimension of a cube response.

    Each dimension represents one of the variables in a cube response. For
    example, a query to cross-tabulate snack-food preference against region
    will have two variables (snack-food preference and region) and will produce
    a two-dimensional (2D) cube response. That cube will have two of these
    dimension objects, which are accessed using
    :attr:`.CrunchCube.dimensions`.
    """

    def __init__(self, dimension_dict, dimension_type, dimension_transforms=None):
        self._unshimmed_dimension_dict = dimension_dict
        self._dimension_type = dimension_type
        self._unshimmed_dimension_transforms_dict = dimension_transforms or {}

    @lazyproperty
    def alias(self) -> Optional[str]:
        """Return the alias for the dimension if it exists, None otherwise."""
        return self._dimension_dict["references"].get("alias")

    @lazyproperty
    def all_elements(self) -> "_AllElements":
        """_AllElements object providing cats or subvars of this dimension.

        Elements in this sequence appear in cube-result order.
        """
        return _AllElements(
            self._dimension_dict["type"],
            self._dimension_transforms_dict,
            self._dimension_type,
        )

    def apply_transforms(self, dimension_transforms) -> "Dimension":
        """Return a new `Dimension` object with `dimension_transforms` applied.

        The new dimension object is the same as this one in all other respects.
        """
        # --- Use unshimmed `._dimension_dict` because we need to re-shim it alongside
        # --- the new dimension_transforms dictionary
        return Dimension(
            self._unshimmed_dimension_dict, self._dimension_type, dimension_transforms
        )

    @lazyproperty
    def description(self) -> str:
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
    def dimension_type(self) -> _DimensionType:
        """Member of DIMENSION_TYPE appropriate to this cube dimension."""
        return self._dimension_type

    @lazyproperty
    def element_aliases(self) -> Tuple[str, ...]:
        """tuple of string element-aliases for each valid element in this dimension.

        Element-aliases appear in the order defined in the cube-result.
        """
        return tuple(e.alias for e in self.valid_elements)

    @lazyproperty
    def element_ids(self) -> Tuple[int, ...]:
        """tuple of int element-id for each valid element in this dimension.

        Element-ids appear in the order defined in the cube-result.
        """
        return tuple(e.element_id for e in self.valid_elements)

    @lazyproperty
    def element_labels(self) -> Tuple[str, ...]:
        """tuple of string element-labels for each valid element in this dimension.

        Element-labels appear in the order defined in the cube-result.
        """
        return tuple(e.label for e in self.valid_elements)

    @lazyproperty
    def hidden_idxs(self) -> Tuple[int, ...]:
        """tuple of int element-idx for each hidden valid element in this dimension.

        An element is hidden when a "hide" transform is applied to it in its transforms
        dict.
        """
        return tuple(
            idx for idx, element in enumerate(self.valid_elements) if element.is_hidden
        )

    @lazyproperty
    def insertion_ids(self) -> Tuple[int, ...]:
        """tuple of int insertion-id for each insertion in this dimension.

        Insertion-ids appear in the order insertions are defined in the dimension.
        """
        return tuple(s.insertion_id for s in self.subtotals)

    @lazyproperty
    def name(self) -> str:
        """str name of this dimension, the empty string ("") if not specified."""
        references = self._dimension_dict["references"]

        def raw_name():
            """Return dimension-name as specified (`None` is not normalized to "")."""
            # --- First authority in cascade is analysis-specific dimension transform.
            # --- None is a legitimate value, indicating suppression of any inherited
            # --- title.
            if "name" in self._dimension_transforms_dict:
                return self._dimension_transforms_dict["name"]
            # --- next authority is base dimension name ---
            if "name" in references:
                return references["name"]
            # --- default is dimension alias ---
            return references.get("alias")

        # ---Normalize None value to "" so return value is always a str and callers
        # ---don't need to deal with multiple possible return value types.
        name = raw_name()
        return name if name else ""

    @lazyproperty
    def numeric_values(self) -> Tuple[Union[int, float], ...]:
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
    def order_spec(self) -> "_OrderSpec":
        """_OrderSpec proxy object for dimension.transforms.order dict from payload."""
        return _OrderSpec(self, self._dimension_transforms_dict)

    @lazyproperty
    def prune(self) -> bool:
        """True if empty elements should be automatically hidden on this dimension."""
        return self._dimension_transforms_dict.get("prune") is True

    @lazyproperty
    def selected_categories(self) -> Tuple[Dict, ...]:
        """List of selected categories specified for this dimension."""
        selected_categories = self._dimension_dict["references"].get(
            "selected_categories"
        )
        return tuple(selected_categories) if selected_categories else ()

    @lazyproperty
    def shape(self) -> int:
        """int count of *all* elements in this dimension, both valid and missing."""
        return len(self.all_elements)

    @lazyproperty
    def smoothing_dict(self) -> Optional[Dict]:
        """Optional dict of smoothing specifications."""
        return self._dimension_transforms_dict.get("smoother") or {}

    @lazyproperty
    def subtotal_aliases(self) -> Tuple[str, ...]:
        """tuple of string element-aliases for each subtotal in this dimension.

        Element-aliases appear in the order defined in the cube-result.
        """
        return tuple(s.alias for s in self.subtotals)

    @lazyproperty
    def subtotal_labels(self) -> Tuple[str, ...]:
        """tuple of string element-labels for each subtotal in this dimension.

        Element-labels appear in the order defined in the cube-result.
        """
        return tuple(s.label for s in self.subtotals)

    @lazyproperty
    def subtotals(self) -> "_Subtotals":
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

    def translate_element_id(self, _id) -> Optional[str]:
        """Optional string that is the translation of various ids to subvariable alias

        This is needed for the opposing dimension's sort by opposing element, because
        when creating a dimension, we don't have access to the other dimension's
        ids to transform it. Therefore, the id for opposing element sort by value
        transforms is not translated at creation time.

        0) If dimension is not a subvariables dimension, return the _id.
        1) If id matches an alias, then just use it.
        2) If id matches a subvariable id, translate to corresponding alias.
        3) If id matches an element id, translate to corresponding alias.
        4) If id can be parsed to int and matches an element id, translate to alias.
        5) If id is int (or can be parsed to int) and can be used as index (eg in range
           0-# of elements), use _id'th alias.
        6) If all of these fail, return None.
        """
        return self._element_id_shim.translate_element_id(_id)

    @lazyproperty
    def valid_elements(self) -> "_ValidElements":
        """_Elements object providing access to non-missing elements.

        Any categories or subvariables representing missing data are excluded
        from the collection; this sequence represents a subset of that
        provided by `.all_elements`.
        """
        return self.all_elements.valid_elements

    @lazyproperty
    def _element_id_shim(self) -> "_ElementIdShim":
        """_ElementIdShim for this Dimension object"""
        return _ElementIdShim(
            self._dimension_type,
            self._unshimmed_dimension_dict,
            self._unshimmed_dimension_transforms_dict,
        )

    @lazyproperty
    def _dimension_dict(self) -> Dict:
        """Copy of dimension dictionary with shimmed `element_id`s"""
        return self._element_id_shim.shimmed_dimension_dict

    @lazyproperty
    def _dimension_transforms_dict(self) -> Dict:
        """Copy of dimension transforms dictionary with shimmed `element_id`s"""
        return self._element_id_shim.shimmed_dimension_transforms_dict


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
    def element_ids(self) -> Tuple[int, ...]:
        """tuple of element-id for each element in collection.

        Element ids appear in the order they occur in the cube response.
        """
        return tuple(element.element_id for element in self._elements)

    @lazyproperty
    def element_idxs(self) -> Tuple[int, ...]:
        """tuple of element-index for each element in collection.

        Element index values represent the position of this element in the
        dimension-dict it came from. In the case of an _AllElements object,
        it will simply be a tuple(range(len(all_elements))).
        """
        return tuple(element.index for element in self._elements)

    def get_by_id(self, element_id: int) -> "_Element":
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
    def _elements_by_id(self) -> Dict:
        """dict mapping each element by its id."""
        return {element.element_id: element for element in self._elements}


class _AllElements(_BaseElements):
    """Sequence of _BaseElement subclass objects for a dimension.

    Each element is either a category or a subvariable.
    """

    def __init__(self, type_dict, dimension_transforms_dict, dimension_type):
        self._type_dict = type_dict
        self._dimension_transforms_dict = dimension_transforms_dict
        self._dimension_type = dimension_type

    @lazyproperty
    def valid_elements(self) -> "_ValidElements":
        """_ValidElements object containing only non-missing elements."""
        return _ValidElements(self._elements, self._dimension_transforms_dict)

    @lazyproperty
    def _element_dicts(self) -> List[Dict]:
        """Sequence of element-dicts for this dimension, taken from cube-result."""
        return (
            self._type_dict["categories"]
            if self._type_dict["class"] == "categorical"
            else self._type_dict["elements"]
        )

    @lazyproperty
    def _elements(self) -> Tuple["_Element", ...]:
        """tuple storing actual sequence of element objects."""
        return tuple(
            _Element(
                element_dict,
                idx,
                _ElementTransforms(element_transforms_dict),
            )
            for (
                idx,
                element_dict,
                element_transforms_dict,
            ) in self._iter_element_makings()
        )

    @lazyproperty
    def _elements_transforms(self) -> Dict:
        """Element transform dict expressed in the dimension transforms expression."""
        return (
            self._shimmed_element_transforms
            if self._dimension_type == DT.MR
            else self._dimension_transforms_dict.get("elements", {})
        )

    def _iter_element_makings(self) -> Iterator[Tuple[int, Dict, Dict]]:
        """Generate tuple of values needed to construct each element object.

        An (idx, element_dict, element_transforms_dict) tuple is generated for each
        element in this dimension, in the order they appear in the cube-result. All
        elements are included (including missing).
        """
        elements_transforms = self._elements_transforms
        for idx, element_dict in enumerate(self._element_dicts):
            # --- convert to string for categorical ids
            element_id = element_dict["id"]
            element_transforms_dict = elements_transforms.get(
                element_id, elements_transforms.get(str(element_id), {})
            )
            yield idx, element_dict, element_transforms_dict

    @lazyproperty
    def _shimmed_element_transforms(self) -> Dict:
        """Element transforms dict for array dimensions.

        To provide consistency with a poorly-defined interface for categorical
        insertions, a client can include a `"hide": true` field in a (complete) copy of
        a variable-based insertion in order to suppress that variable-based insertion.

        For the array case, these need to be translated to a "hide" transform on the
        subvariable element because such an insertion becomes a derived subvariable just
        like the other subvariables in the dimension.
        """
        # --- currently an inserted-subvariable can only be identified by name, there is
        # --- no alias for an inserted-subvariable and it does not receive a "normal"
        # --- element.id like "0001".
        hidden_insertion_names = tuple(
            insertion["name"]
            for insertion in self._dimension_transforms_dict.get("insertions", [])
            if insertion.get("hide", False)
        )

        # --- however, the hide-transform must be identified by element-id, so we need a
        # --- mapping of insertion-name to element-id
        element_id_from_name = {
            element["value"]["id"]: element["id"] for element in self._element_dicts
        }

        # --- merge hide transforms with (a copy of) the existing element transforms ---
        hidden_transforms = {
            element_id_from_name[name]: {"hide": True}
            for name in hidden_insertion_names
        }
        element_transforms = self._dimension_transforms_dict.get("elements", {})
        return {**hidden_transforms, **element_transforms}


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
    def _elements(self) -> Tuple["_Element", ...]:
        """tuple containing actual sequence of element objects."""
        return tuple(element for element in self._all_elements if not element.missing)


class _ElementIdShim:
    """Object used to replace element ids with alias for subvariables.

    We want to move to a world where elements on a subvariables dimension are
    identified by their alias, but right now the "element_id" from zz9 is
    an index, and the transforms have several different ways to refer to
    subvariables.

    Types of identifiers for subvariables (and derived insertions):

    * "element_id": Stored in the cube result as the object name in
      `dimensions[i].type.elements[j].id`. For subvariables, zz9 currently puts
      the index integer here. Long term zz9 may change this to the the alias.
    * "subvariable_id": Subvariables have an id stored in
      `dimensions[i].type.elements[j].value.id`, generally this is a 4 digit,
      0-padded index of the subvariable when it was first created (eg "0001",
      "0002", ...), though it is not required to be. For derived insertions,
      currently the name is used here.
    * "alias": Subvariables also have an alias that identifies them. It is stored
      in `dimensions[i].type.elements[j].value.
    """

    def __init__(self, dimension_type, dimension_dict, dimension_transforms_dict):
        self._dimension_type = dimension_type
        self._dimension_dict = dimension_dict
        self._dimension_transforms_dict = dimension_transforms_dict

    @lazyproperty
    def shimmed_dimension_dict(self) -> Dict:
        """Copy of dimension dictionary with shimmed `element_id`s

        We want to move to a world where elements on a subvariables dimension are
        identified by their alias, but right now the "element_id" from zz9 is
        an index for subvariables.
        """
        shim = copy.deepcopy(self._dimension_dict)

        # --- Leave non-subvariable dimension types alone, as they don't have
        # --- subvariable aliases to use, and category ids are already the main way
        # --- we identify elements on categorical dimensions (and this is correct).
        if self._dimension_type not in DT.ARRAY_TYPES:
            return shim

        # --- Replace element ids with the alias
        for idx, alias in enumerate(self._subvar_aliases):
            shim["type"]["elements"][idx]["id"] = alias

        return shim

    @lazyproperty
    def shimmed_dimension_transforms_dict(self) -> Dict:
        """Copy of dimension transforms dictionary with shimmed `element_id`s

        We want to move to a world where elements on a subvariables dimension are
        identified by their alias, but right now the "element_id" from zz9 is
        simply the subvariable's (unstable) cardinal position in subvariables
        sequence. Different parts of the transforms have several different ways
        to refer to subvariables.

        Types of identifiers for subvariables (and derived insertions):
        - "element_id": Stored in the cube result as the object name in
          `dimensions[i].type.elements[j].id`. For subvariables, zz9 currently puts
          the index integer here. Long term zz9 may change this to the the alias.
        - "subvariable_id": Subvariables have an id stored in
          `dimensions[i].type.elements[j].value.id`, generally this is a 4 digit,
          0-padded index of the subvariable when it was first created (eg "0001",
          "0002", ...), though it is not required to be. For derived insertions,
          currently the name is used here.
        - "alias": Subvariables also have an alias that identifies them. It is stored
          in `dimensions[i].type.elements[j].value.references.alias`.
        """
        shim = copy.deepcopy(self._dimension_transforms_dict)

        # --- Leave non-subvariable dimension types alone, as they don't have
        # --- subvariable aliases to use, and category ids are already the main way
        # --- we identify elements on categorical dimensions (and this is correct).
        if self._dimension_type not in DT.ARRAY_TYPES:
            return shim

        # --- Replace element transform ids with the alias
        if "elements" in shim:
            shim["elements"] = self._replaced_element_transforms(shim["elements"])

        # --- Translate explicit order element ids if present
        if shim.get("order", {}).get("element_ids") is not None:
            shim["order"]["element_ids"] = self._replaced_order_element_ids(
                shim["order"]["element_ids"]
            )

        # --- sort-by-value on the opposing dimension also refers to element ids, but
        # --- the ids refer to the opposing dimension, so do the translation later on.
        # --- This is a little unfortunate, because this means that the ids in this shim
        # --- version of the dimension transforms are inconsistent. But it feels easier
        # --- than forcing the dimensions to be aware of other dimensions.

        return shim

    def translate_element_id(self, _id) -> Optional[str]:
        """Optional string that is the translation of various ids to subvariable alias

        0) If dimension is not a subvariables dimension, return the _id.
        1) If id matches an alias, then just use it.
        2) If id matches an element id, translate to corresponding alias.
        3) If id matches a subvariable id, translate to corresponding alias.
        4) If id can be parsed to int and matches an element id, translate to alias.
        5) If id is int (or can be parsed to int) and can be used as index (eg in range
           0-# of elements), use _id'th alias.
        6) If all of these fail, return None.
        """
        if self._dimension_type not in DT.ARRAY_TYPES:
            return _id

        if _id in self._subvar_aliases:
            return _id
        if _id in self._raw_element_ids:
            return self._subvar_aliases[self._raw_element_ids.index(_id)]
        if _id in self._subvar_ids:
            return self._subvar_aliases[self._subvar_ids.index(_id)]

        try:
            _id = int(_id)
            # --- If successfully converted to int, try raw element ids again
            if _id in self._raw_element_ids:
                return self._subvar_aliases[self._raw_element_ids.index(_id)]
        except ValueError:
            return None

        if _id >= 0 and _id < len(self._subvar_aliases):
            return self._subvar_aliases[_id]

        return None

    @lazyproperty
    def _raw_element_ids(self) -> Tuple[Union[int, str], ...]:
        """tuple of int or string element ids, as they appear in cube result

        These are "raw" because they refer to the element ids before they've been
        replaced with the alias for subvariables in the `._shimmed_dimension_dict`.
        """
        return tuple(
            element["id"] for element in self._dimension_dict["type"]["elements"]
        )

    def _replaced_element_transforms(self, element_transforms) -> Dict:
        """Replace the dictionary keys of element transforms with aliases

        The element transforms identify which element they refer to by their key in
        the element_transforms object. Before it is shimmed, this can identify them
        in many different ways. The shim replaces these with the alias.
        """
        # --- The name of the element transform object is a string. This string is
        # --- assumed to refer to the subvariable id, unless there is a "key", which
        # --- can be "alias", in which case it refers to the alias, or if no
        # --- subvariable is found with the id, in which case it's assumed to be the
        # --- element_id
        # --- TODO: This logic about "key" is not supported by the validation
        # --- on the deck schema, so maybe we should remove it?
        key = element_transforms.get("key")
        if key == "alias":
            # --- Already keyed by alias, no changes needed
            return element_transforms

        old_keys = tuple(element_transforms.keys())

        if key == "subvar_id":
            # --- translate from subvariable id
            new_keys = tuple(
                self._subvar_aliases[self._subvar_ids.index(_id)]
                if _id in self._subvar_ids
                else None
                for _id in old_keys
            )
        else:
            # --- Otherwise use usual translation logic
            new_keys = tuple(self.translate_element_id(_id) for _id in old_keys)

        return {
            nkey: element_transforms[old_keys[i]]
            for i, nkey in enumerate(new_keys)
            if nkey is not None
        }

    def _replaced_order_element_ids(self, element_ids) -> List[Optional[str]]:
        """Replace the list of element ids with a list of aliases

        The explicit order transform includes a list of ids that can be specified in
        many different ways, this translate them to the subvariable aliases.
        """
        return [self.translate_element_id(_id) for _id in element_ids]

    @lazyproperty
    def _subvar_aliases(self) -> Tuple[str, ...]:
        """tuple of str alias for each element of a subvariable dimension

        Fall back to the `element_id` if the alias doesn't exist (this happens in one
        fixture, but I don't think it can happen in production anymore.)
        """
        return tuple(
            element.get("value", {}).get("references", {}).get("alias", element["id"])
            for element in self._dimension_dict["type"]["elements"]
        )

    @lazyproperty
    def _subvar_ids(self) -> Tuple[Union[int, str], ...]:
        """tuple of str subvariable id for each element of a subvariable dimension

        Only applicable to subvariables dimension (will raise KeyError if not).
        """
        return tuple(
            element["value"]["id"]
            for element in self._dimension_dict["type"]["elements"]
        )


class _Element:
    """A category or subvariable of a dimension.

    This object resolves the transform cascade for element-level transforms.
    """

    def __init__(self, element_dict, index, element_transforms):
        self._element_dict = element_dict
        self._index = index
        self._element_transforms = element_transforms

    @lazyproperty
    def alias(self) -> str:
        """str display-alias for this element."""
        return self._str_representation_for("alias")

    @lazyproperty
    def anchor(self) -> Optional[Union[str, dict]]:
        """Optional str or dict defining the anchor for derived elements"""
        if not self.derived:
            return None

        return self._element_dict.get("value", {}).get("references", {}).get("anchor")

    @lazyproperty
    def element_id(self) -> int:
        """int identifier for this category or subvariable."""
        return self._element_dict["id"]

    @lazyproperty
    def derived(self) -> bool:
        """True if element is derived, False otherwise.

        Multiple Response subvariable insertions are considered derived elements.
        """
        value = self._element_dict.get("value")
        if not value or not isinstance(value, dict):
            return False
        return value.get("derived", False)

    @lazyproperty
    def fill(self) -> str:
        """str RGB color like "#af032d" or None if not specified.

        A value of None indicates the default fill should be used for this element.
        A str value must be a hash character ("#") followed by six hexadecimal digits.
        Three-character color contractions (like "#D07") are not valid.
        """
        # ---first authority is fill transform in element transforms. The base value is
        # ---the only prior authority and that is None (use default fill color).
        return self._element_transforms.fill

    @lazyproperty
    def index(self) -> int:
        """int offset at which this element appears in dimension.

        This position is based upon the document position of this element in
        the cube response. No adjustment for missing elements is made.
        """
        return self._index

    @lazyproperty
    def is_hidden(self) -> bool:
        """True if this element is explicitly hidden in this analysis."""
        # ---first authority is hide transform in element transforms---
        # ---default is not hidden, there is currently no prior-level hide transform---
        return {True: True, False: False, None: False}[self._element_transforms.hide]

    @lazyproperty
    def label(self) -> str:
        """str display-name for this element.

        This value is the empty string when no value has been specified or display of
        the name has been suppressed.
        """
        return self._str_representation_for("name")

    @lazyproperty
    def missing(self) -> bool:
        """True if this element represents missing data.

        False if this category or subvariable represents valid (collected)
        data.
        """
        return bool(self._element_dict.get("missing"))

    @lazyproperty
    def numeric_value(self) -> Union[int, float]:
        """Numeric value assigned to element by user, np.nan if absent."""
        numeric_value = self._element_dict.get("numeric_value")
        return np.nan if numeric_value is None else numeric_value

    def _str_representation_for(self, key: str) -> str:
        """return str representation for this element of a given key (`alias` or `name`)

        This method handles elements for variables of all types, including
        categorical, array (subvariable), numeric, datetime and text.
        """
        # ---first authority is transform in element transforms---
        value = getattr(self._element_transforms, key) if key == "name" else None
        if value is not None:
            return value if value else ""

        # ---otherwise base-name/alias from element-dict is used according to the key---
        element_dict = self._element_dict

        # ---category elements have a name/alias item according to the key---
        if key in element_dict:
            value = element_dict[key]
            return value if value else ""

        # ---other types are more complicated---
        value = element_dict.get("value")
        type_value = type(value).__name__

        if type_value == "NoneType":
            return ""

        if type_value == "list":
            # ---like '10-15' or 'A-F'---
            return "-".join([str(item) for item in value])

        if type_value in ("float", "int"):
            return str(value)

        if type_value in ("str", "unicode"):
            return value

        # ---For CA and MR subvar dimensions---
        return value.get("references", {}).get(key) or ""


class _ElementTransforms:
    """A value object providing convenient access to transforms for a single element."""

    def __init__(self, element_transforms_dict):
        self._element_transforms_dict = element_transforms_dict

    @lazyproperty
    def fill(self) -> Optional[str]:
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
    def hide(self) -> Optional[bool]:
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
    def name(self) -> Optional[str]:
        """Optional str display-name for this element (None if not specified)."""
        # ---if "name": element is omitted, no transform is specified---
        if "name" not in self._element_transforms_dict:
            return None
        # ---otherwise normalize value to str, with an explicit value of None, [], 0,
        # ---etc. becoming the empty string ("").
        name = self._element_transforms_dict["name"]
        return str(name) if name else ""


class _OrderSpec:
    """Value object providing convenient access to details of an order transform."""

    def __init__(self, dimension, dimension_transforms_dict):
        self._dimension = dimension
        self._dimension_transforms_dict = dimension_transforms_dict

    @lazyproperty
    def bottom_fixed_ids(self) -> Tuple[int, ...]:
        """Tuple of each element-id appearing in the fixed.bottom field of order dict.

        The element-ids appear in the order specified in the "bottom" fixed.bottom
        field.
        """
        return tuple(self._order_dict.get("fixed", {}).get("bottom", []))

    @lazyproperty
    def collation_method(self) -> CM:
        """Member of COLLATION_METHOD specifying ordering of dimension elements."""
        method_keyword = self._order_dict.get("type")
        if method_keyword is None or not CM.has_value(method_keyword):
            return CM.PAYLOAD_ORDER
        return CM(method_keyword)

    @lazyproperty
    def descending(self) -> bool:
        """True if sort direction is descending, False otherwise."""
        return self._order_dict.get("direction", "descending") != "ascending"

    @lazyproperty
    def element_id(self) -> int:
        """int element id appearing in an order transform.

        Raises KeyError if the transform dict does not contain an "element_id" field.
        Note that not all order types use this field but it is a required field in all
        those that do.
        """
        return self._order_dict["element_id"]

    @lazyproperty
    def element_ids(self) -> Tuple[int, ...]:
        """tuple of int each element id appearing in an explicit order transform.

        This value is `()` if no "element_ids": field is present.
        """
        return tuple(self._order_dict.get("element_ids") or [])

    @lazyproperty
    def insertion_id(self) -> int:
        """int insertion-id in the "insertion_id" field of the transform dict.

        Raises KeyError if this transform dict does not contain an "insertion_id" field.
        """
        return self._order_dict["insertion_id"]

    @lazyproperty
    def marginal(self) -> MARGINAL:
        """Member of enums.MARGINAL corresponding to "marginal": field in order transform.

        Raises KeyError if the order dict has no "marginal": field and ValueError if the
        value in that field is not a recognized marginal keyword. Note that not all order
        types use the "marginal": field.
        """
        return MARGINAL(self.marginal_keyname)

    @lazyproperty
    def marginal_keyname(self) -> str:
        """str value of "marginal": field in order transform.

        Raises KeyError if the order dict has no "marginal": field. Note that not all
        order types use the "measure": field, but it is a required field in all that do.
        """
        return self._order_dict["marginal"]

    @lazyproperty
    def measure(self) -> MEASURE:
        """Member of enums.MEASURE corresponding to "measure": field in order transform.

        Raises KeyError if the order dict has no "measure": field and ValueError if the
        value in that field is not a recognized measure keyword. Note that not all order
        types use the "measure": field.
        """
        return MEASURE(self.measure_keyname)

    @lazyproperty
    def measure_keyname(self) -> str:
        """str value of "measure": field in order transform.

        Raises KeyError if the order dict has no "measure": field. Note that not all
        order types use the "measure": field, but it is a required field in all that do.
        Note also that this property is required for the "univariate_measure" sort type
        because its measure keywords do not correspond directly to members of the
        MEASURE enumeration.
        """
        return self._order_dict["measure"]

    @lazyproperty
    def top_fixed_ids(self) -> Tuple[int, ...]:
        """Tuple of each element-id appearing in the fixed.top field of order dict.

        The element-ids appear in the order specified in the "top" fixed.top
        field.
        """
        return tuple(self._order_dict.get("fixed", {}).get("top", []))

    @lazyproperty
    def _order_dict(self) -> Dict:
        """dict dimension.transforms.order field parsed from JSON payload.

        Value is `{}` if no "order": field is present.
        """
        return self._dimension_transforms_dict.get("order") or {}


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
            if not {"anchor", "name"}.issubset(insertion_dict.keys()):
                continue

            # ---use "new style" kwargs defining positive terms if available---
            # ---but if missing, use "old style" args defining positive terms---
            positive = insertion_dict.get("kwargs", {}).get(
                "positive"
            ) or insertion_dict.get("args", [])

            negative = insertion_dict.get("kwargs", {}).get("negative", [])

            # ---must have positive or negative elements---
            if not (positive or negative):
                continue

            # ---skip if doesn't reference at least one non-missing element---
            if not self._element_ids.intersection(positive + negative):
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


class _Subtotal:
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
    def alias(self) -> str:
        """str display alias for this subtotal."""
        alias = self._subtotal_dict.get("alias")
        return alias if alias else ""

    @lazyproperty
    def anchor(self) -> Union[int, str]:
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
    def addend_ids(self) -> Tuple[int, ...]:
        """tuple of int ids of elements contributing to this subtotal.

        Any element id not present in the dimension or present but
        representing missing data is excluded.
        """
        # ---Prefer positive "kwargs" over "args" so we can migrate---
        positive = self._subtotal_dict.get("kwargs", {}).get(
            "positive"
        ) or self._subtotal_dict.get("args", [])

        return tuple(arg for arg in positive if arg in self._valid_elements.element_ids)

    @lazyproperty
    def addend_idxs(self) -> np.ndarray:
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
            dtype=np.int64,  # force int so it can be used as index even if empty
        )

    @lazyproperty
    def fill(self) -> str:
        """str RGB color like "#af032d" or None if not specified.

        A value of None indicates the default fill should be used for this element.
        A str value must be a hash character ("#") followed by six hexadecimal digits.
        Three-character color contractions (like "#D07") are not valid.
        """
        # ---first authority is fill transform in element transforms. The base value is
        # ---the only prior authority and that is None (use default fill color).
        return self._subtotal_dict.get("fill", None)

    @lazyproperty
    def insertion_id(self) -> int:
        """int unique identifier of this subtotal within this dimension's insertions."""
        return self._subtotal_dict.get("id", self._fallback_insertion_id)

    @lazyproperty
    def is_difference(self) -> bool:
        """True if a subtotal is a difference, False otherwise."""
        return bool(self.subtrahend_ids)

    @lazyproperty
    def label(self) -> str:
        """str display name for this subtotal, suitable for use as label."""
        name = self._subtotal_dict.get("name")
        return name if name else ""

    @lazyproperty
    def subtrahend_ids(self) -> Tuple[int, ...]:
        """tuple of int ids of elements of negative part of the subtotal.

        Any element id not present in the dimension or present but
        representing missing data is excluded.
        """
        return tuple(
            arg
            for arg in self._subtotal_dict.get("kwargs", {}).get("negative", [])
            if arg in self._valid_elements.element_ids
        )

    @lazyproperty
    def subtrahend_idxs(self) -> np.ndarray:
        """ndarray of int base-element offsets contributing to the negative part of subtotal.

        Suitable for directly indexing a numpy array object (such as base values or
        margin) to extract the addend values for this subtotal.
        """
        return np.fromiter(
            (
                idx
                for idx, vector in enumerate(self._valid_elements)
                if vector.element_id in self.subtrahend_ids
            ),
            dtype=np.int64,  # force int so it can be used as index even if empty
        )
