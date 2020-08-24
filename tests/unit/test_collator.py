# encoding: utf-8

"""Unit test suite for `cr.cube.collator` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import numpy as np
import pytest

from cr.cube.collator import (
    _BaseCollator,
    _BaseAnchoredCollator,
    _BaseSortByValueCollator,
    ExplicitOrderCollator,
    MarginalCollator,
    OpposingElementCollator,
    OpposingSubtotalCollator,
    PayloadOrderCollator,
)
from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.matrix import _BaseInsertedVector, _CategoricalVector

from ..unitutil import (
    ANY,
    call,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class Describe_BaseCollator(object):
    """Unit-test suite for `cr.cube.collator._BaseCollator` object."""

    def it_provides_access_to_the_dimension_element_ids_to_help(self, dimension_):
        dimension_.element_ids = (42, 24, 1, 6)
        collator = _BaseCollator(dimension_)

        element_ids = collator._element_ids

        assert element_ids == (42, 24, 1, 6)

    def it_provides_access_to_the_order_transform_dict_to_help(self, dimension_):
        dimension_.order_dict = {"order": "dict"}
        collator = _BaseCollator(dimension_)

        order_dict = collator._order_dict

        assert order_dict == {"order": "dict"}

    @pytest.mark.parametrize(
        "subtotals, dimension_type, expected_value",
        (
            ((), DT.CAT, ()),
            ((), DT.MR, ()),
            (("sub", "tot", "als"), DT.CAT, ("sub", "tot", "als")),
            (("sub", "tot", "als"), DT.CA_SUBVAR, ()),
            (("sub", "tot", "als"), DT.MR, ()),
        ),
    )
    def it_provides_access_to_the_dimension_subtotals_to_help(
        self, dimension_, subtotals, dimension_type, expected_value
    ):
        dimension_.dimension_type = dimension_type
        dimension_.subtotals = subtotals
        collator = _BaseCollator(dimension_)

        subtotals = collator._subtotals

        assert subtotals == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_BaseAnchoredCollator(object):
    """Unit-test suite for `cr.cube.collator._BaseAnchoredCollator` object."""

    def it_provides_an_interface_classmethod(self, request, dimension_):
        _init_ = initializer_mock(request, _BaseAnchoredCollator)
        _display_order_ = property_mock(
            request, _BaseAnchoredCollator, "_display_order"
        )
        _display_order_.return_value = (-3, 0, 1, -2, 2, 3, -1)

        display_order = _BaseAnchoredCollator.display_order(dimension_)

        _init_.assert_called_once_with(ANY, dimension_)
        assert display_order == (-3, 0, 1, -2, 2, 3, -1)

    @pytest.mark.parametrize(
        "base_orderings, insertion_orderings, expected_value",
        (
            # --- base-values but no insertions ---
            (((0, 0), (1, 1), (2, 2)), (), (0, 1, 2)),
            # --- insertions but no base-values (not expected) ---
            ((), ((sys.maxsize, -3), (0, -2), (sys.maxsize, -1)), (-2, -3, -1)),
            # --- both base-values and insertions ---
            (
                ((0, 0), (1, 1), (2, 2), (3, 3)),
                ((0, -3), (2, -2), (sys.maxsize, -1)),
                (-3, 0, 1, -2, 2, 3, -1),
            ),
        ),
    )
    def it_computes_the_display_order_to_help(
        self, request, base_orderings, insertion_orderings, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_insertion_orderings",
            return_value=insertion_orderings,
        )
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_base_element_orderings",
            return_value=base_orderings,
        )
        collator = _BaseAnchoredCollator(None)

        display_order = collator._display_order

        assert display_order == expected_value

    @pytest.mark.parametrize(
        "element_order_descriptors, expected_value",
        (
            # --- no elements in dimension (not an expected case) ---
            ((), ()),
            # --- 1 element ---
            (((0, 0, 6),), ((0, 0),)),
            # --- 3 elements in dimension ---
            (((0, 2, 18), (1, 0, 12), (2, 1, 5)), ((0, 2), (1, 0), (2, 1))),
        ),
    )
    def it_computes_the_base_element_orderings_to_help(
        self, request, element_order_descriptors, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_element_order_descriptors",
            return_value=element_order_descriptors,
        )
        collator = _BaseAnchoredCollator(None)

        base_element_orderings = collator._base_element_orderings

        assert base_element_orderings == expected_value

    @pytest.mark.parametrize(
        "element_order_descriptors, expected_value",
        (
            # --- no elements in dimension (not an expected case) ---
            ((), {}),
            # --- 1 element ---
            (((0, 0, 6),), {6: 0}),
            # --- 3 elements in dimension ---
            (((0, 2, 18), (1, 0, 12), (2, 1, 5)), {18: 0, 12: 1, 5: 2}),
        ),
    )
    def it_maps_the_element_ids_to_their_position_to_help(
        self, request, element_order_descriptors, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_element_order_descriptors",
            return_value=element_order_descriptors,
        )
        collator = _BaseAnchoredCollator(None)

        element_positions_by_id = collator._element_positions_by_id

        assert element_positions_by_id == expected_value

    @pytest.mark.parametrize(
        "insertion_positions, expected_value",
        (
            # --- no inserted-subtotals in dimension ---
            ((), ()),
            # --- 1 insertion ---
            ((6,), ((6, -1),)),
            # --- 3 insertions ---
            ((7, sys.maxsize, 0), ((7, -3), (sys.maxsize, -2), (0, -1))),
        ),
    )
    def it_computes_the_insertion_orderings_to_help(
        self, request, insertion_positions, expected_value
    ):
        subtotals_ = tuple(
            instance_mock(request, _Subtotal) for _ in insertion_positions
        )
        property_mock(
            request, _BaseAnchoredCollator, "_subtotals", return_value=subtotals_
        )
        _insertion_position_ = method_mock(
            request,
            _BaseAnchoredCollator,
            "_insertion_position",
            side_effect=iter(insertion_positions),
        )
        collator = _BaseAnchoredCollator(None)

        insertion_orderings = collator._insertion_orderings

        assert _insertion_position_.call_args_list == [
            call(collator, s) for s in subtotals_
        ]
        assert insertion_orderings == expected_value

    @pytest.mark.parametrize(
        "anchor, expected_value",
        (
            ("top", 0),
            ("bottom", sys.maxsize),
            (666, sys.maxsize),
            (1, 1),
            (2, 2),
            (3, 3),
        ),
    )
    def it_can_compute_the_insertion_position_for_a_subtotal_to_help(
        self, request, anchor, expected_value
    ):
        property_mock(
            request,
            _BaseAnchoredCollator,
            "_element_positions_by_id",
            return_value={1: 0, 2: 1, 3: 2},
        )
        subtotal_ = instance_mock(request, _Subtotal, anchor=anchor)
        collator = _BaseAnchoredCollator(None)

        insertion_position = collator._insertion_position(subtotal_)

        assert insertion_position == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_BaseSortByValueCollator(object):
    """Unit-test suite for `cr.cube.collator._BaseSortByValueCollator` object."""

    def it_computes_the_display_order_to_help(
        self, request, _top_exclusion_idxs_prop_, _bottom_exclusion_idxs_prop_
    ):
        _top_exclusion_idxs_prop_.return_value = (3,)
        _bottom_exclusion_idxs_prop_.return_value = (7, 8)
        property_mock(
            request, _BaseSortByValueCollator, "_top_subtotal_idxs", return_value=(1, 2)
        )
        property_mock(
            request, _BaseSortByValueCollator, "_body_idxs", return_value=(4, 5, 6)
        )
        property_mock(
            request,
            _BaseSortByValueCollator,
            "_bottom_subtotal_idxs",
            return_value=(9,),
        )
        collator = _BaseSortByValueCollator(None)

        display_order = collator._display_order

        assert display_order == (1, 2, 3, 4, 5, 6, 7, 8, 9)

    @pytest.mark.parametrize(
        "top_excl_idxs, bottom_excl_idxs, descending, element_values, expected_value",
        (
            # --- ascending sort ---
            ((), (), False, (8.0, 2.0, 4.0, 1.0), (3, 1, 2, 0)),
            # --- ascending with top exclusions (which therefore do not appear ---
            ((2, 1), (), False, (8.0, 2.0, 4.0, 1.0), (3, 0)),
            # --- descending sort ---
            ((), (), True, (8.0, 2.0, 4.0, 1.0), (0, 2, 1, 3)),
            # --- descending with bottom exclusions ---
            ((), (1, 0), True, (8.0, 2.0, 4.0, 1.0), (2, 3)),
            # --- descending with both kinds of exclusion ---
            ((0,), (3,), True, (8.0, 2.0, 4.0, 1.0), (2, 1)),
        ),
    )
    def it_computes_the_sorted_body_idxs_to_help(
        self,
        request,
        _top_exclusion_idxs_prop_,
        _bottom_exclusion_idxs_prop_,
        _descending_prop_,
        top_excl_idxs,
        bottom_excl_idxs,
        descending,
        element_values,
        expected_value,
    ):
        """Body-idxs are for elements that are not subtotals and not excluded."""
        _top_exclusion_idxs_prop_.return_value = top_excl_idxs
        _bottom_exclusion_idxs_prop_.return_value = bottom_excl_idxs
        _descending_prop_.return_value = descending
        property_mock(
            request,
            _BaseSortByValueCollator,
            "_element_values",
            return_value=element_values,
        )
        collator = _BaseSortByValueCollator(None)

        body_idxs = collator._body_idxs

        assert body_idxs == expected_value

    def it_knows_the_bottom_exclusion_idxs_to_help(self, _iter_exclusion_idxs_):
        _iter_exclusion_idxs_.return_value = (n for n in (4, 0, 5, 2))
        collator = _BaseSortByValueCollator(None)

        bottom_exclusion_idxs = collator._bottom_exclusion_idxs

        _iter_exclusion_idxs_.assert_called_once_with(collator, "bottom")
        assert bottom_exclusion_idxs == (4, 0, 5, 2)

    @pytest.mark.parametrize(
        "descending, subtotal_idxs, expected_value",
        (
            # --- ascending sort, all subtotals at bottom ---
            (False, (-3, -1, -2), (-3, -1, -2)),
            # --- descending sort, no subtotals at bottom ---
            (True, (-3, -1, -2), ()),
        ),
    )
    def it_knows_the_bottom_subtotal_idxs_to_help(
        self,
        _descending_prop_,
        _subtotal_idxs_prop_,
        descending,
        subtotal_idxs,
        expected_value,
    ):
        _descending_prop_.return_value = descending
        _subtotal_idxs_prop_.return_value = subtotal_idxs
        collator = _BaseSortByValueCollator(None)

        bottom_subtotal_idxs = collator._bottom_subtotal_idxs

        assert bottom_subtotal_idxs == expected_value

    @pytest.mark.parametrize(
        "order_dict, expected_value",
        (
            ({"direction": "ascending"}, False),
            ({"direction": "descending"}, True),
            ({}, True),
            ({"direction": None}, True),
            ({"direction": 42}, True),
            ({"direction": False}, True),
            ({"direction": "foobar"}, True),
        ),
    )
    def it_knows_whether_the_sort_direction_is_descending(
        self, _order_dict_prop_, order_dict, expected_value
    ):
        """Otherwise, sort direction is ascending."""
        _order_dict_prop_.return_value = order_dict
        collator = _BaseSortByValueCollator(None)

        descending = collator._descending

        assert descending == expected_value

    def it_provides_a_mapping_of_element_id_to_element_idx_to_help(self, request):
        property_mock(
            request, _BaseSortByValueCollator, "_element_ids", return_value=(1, 2, 3)
        )
        collator = _BaseSortByValueCollator(None)

        element_idxs_by_id = collator._element_idxs_by_id

        assert element_idxs_by_id == {1: 0, 2: 1, 3: 2}

    @pytest.mark.parametrize(
        "top_or_bottom, order_dict, expected_value",
        (
            ("top", {}, ()),
            ("bottom", {"exclude": {}}, ()),
            ("bottom", {"exclude": {"foobar": [3, 4]}}, ()),
            ("top", {"exclude": {"top": [2, 3]}}, (1, 2)),
            ("bottom", {"exclude": {"top": [2, 3]}}, ()),
            ("top", {"exclude": {"bottom": [2, 3]}}, ()),
            ("bottom", {"exclude": {"bottom": [2]}}, (1,)),
            ("top", {"exclude": {"top": [2, 5, 3]}}, (1, 2)),
        ),
    )
    def it_can_generate_the_exclusion_idxs_for_top_or_bottom_to_help(
        self, request, _order_dict_prop_, top_or_bottom, order_dict, expected_value
    ):
        property_mock(
            request,
            _BaseSortByValueCollator,
            "_element_idxs_by_id",
            return_value={1: 0, 2: 1, 3: 2, 4: 3},
        )
        _order_dict_prop_.return_value = order_dict
        collator = _BaseSortByValueCollator(None)

        exclusion_idx_iterator = collator._iter_exclusion_idxs(top_or_bottom)

        assert tuple(exclusion_idx_iterator) == expected_value

    @pytest.mark.parametrize(
        "measure_keyword, expected_value",
        (("unweighted_count", "unweighted_counts"), ("count", "counts")),
    )
    def it_maps_the_measure_keyword_to_its_vector_property_name_to_help(
        self, request, _order_dict_prop_, measure_keyword, expected_value
    ):
        _order_dict_prop_.return_value = {"measure": measure_keyword}
        collator = _BaseSortByValueCollator(None)

        measure_propname = collator._measure_propname

        assert measure_propname == expected_value

    @pytest.mark.parametrize(
        "descending, subtotal_values, expected_value",
        (
            # --- ascending sort ---
            (False, (8.0, 2.0, 4.0), (-2, -1, -3)),
            (False, (8.0, np.nan, 4.0, np.nan), (-2, -4, -3, -1)),
            # --- descending sort ---
            (True, (4.0, 2.0, 8.0), (-1, -3, -2)),
            (True, (8.0, np.nan, 4.0, np.nan), (-4, -2, -3, -1)),
        ),
    )
    def it_computes_subtotal_idxs_from_subclass_values_to_help(
        self, request, _descending_prop_, descending, subtotal_values, expected_value
    ):
        """Sorting, neg-idx generation and NaN-handling are common across subclasses."""
        _descending_prop_.return_value = descending
        property_mock(
            request,
            _BaseSortByValueCollator,
            "_subtotal_values",
            return_value=subtotal_values,
        )
        collator = _BaseSortByValueCollator(None)

        subtotal_idxs = collator._subtotal_idxs

        assert subtotal_idxs == expected_value

    def it_knows_the_top_exclusion_idxs_to_help(self, _iter_exclusion_idxs_):
        _iter_exclusion_idxs_.return_value = (n for n in (3, 1, 4, 2))
        collator = _BaseSortByValueCollator(None)

        top_exclusion_idxs = collator._top_exclusion_idxs

        _iter_exclusion_idxs_.assert_called_once_with(collator, "top")
        assert top_exclusion_idxs == (3, 1, 4, 2)

    @pytest.mark.parametrize(
        "descending, subtotal_idxs, expected_value",
        (
            # --- ascending sort, no subtotals at top ---
            (False, (-3, -1, -2), ()),
            # --- descending sort, all subtotals at top ---
            (True, (-3, -1, -2), (-3, -1, -2)),
        ),
    )
    def it_knows_the_top_subtotal_idxs_to_help(
        self,
        _descending_prop_,
        _subtotal_idxs_prop_,
        descending,
        subtotal_idxs,
        expected_value,
    ):
        _descending_prop_.return_value = descending
        _subtotal_idxs_prop_.return_value = subtotal_idxs
        collator = _BaseSortByValueCollator(None)

        top_subtotal_idxs = collator._top_subtotal_idxs

        assert top_subtotal_idxs == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _bottom_exclusion_idxs_prop_(self, request):
        return property_mock(
            request, _BaseSortByValueCollator, "_bottom_exclusion_idxs"
        )

    @pytest.fixture
    def _descending_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_descending")

    @pytest.fixture
    def _iter_exclusion_idxs_(self, request):
        return method_mock(request, _BaseSortByValueCollator, "_iter_exclusion_idxs")

    @pytest.fixture
    def _order_dict_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_order_dict")

    @pytest.fixture
    def _subtotal_idxs_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_subtotal_idxs")

    @pytest.fixture
    def _top_exclusion_idxs_prop_(self, request):
        return property_mock(request, _BaseSortByValueCollator, "_top_exclusion_idxs")


class DescribeExplicitOrderCollator(object):
    """Unit-test suite for `cr.cube.collator.ExplicitOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, element_id_order, expected_value",
        (
            ((), [], ()),
            ((1, 2, 3), [2, 3, 1], ((0, 1, 2), (1, 2, 3), (2, 0, 1))),
            # --- element not mentioned in transform appears at end in payload order ---
            ((1, 2, 3), [], ((0, 0, 1), (1, 1, 2), (2, 2, 3))),
            ((1, 2, 3), [2], ((0, 1, 2), (1, 0, 1), (2, 2, 3))),
            # --- id appearing in transform but not dimension is ignored ---
            ((), [3, 2, 1], ()),
            ((1, 2, 3), [2, 4, 1], ((0, 1, 2), (1, 0, 1), (2, 2, 3))),
            # --- id repeated in transform is only used on first encounter ---
            ((1, 2, 3), [2, 2, 2], ((0, 1, 2), (1, 0, 1), (2, 2, 3))),
            ((1, 2, 3), [3, 1, 3], ((0, 2, 3), (1, 0, 1), (2, 1, 2))),
        ),
    )
    def it_computes_the_element_order_descriptors_to_help(
        self, request, element_ids, element_id_order, expected_value
    ):
        property_mock(
            request, ExplicitOrderCollator, "_element_ids", return_value=element_ids
        )
        property_mock(
            request,
            ExplicitOrderCollator,
            "_order_dict",
            return_value={"element_ids": element_id_order},
        )
        collator = ExplicitOrderCollator(None)

        element_order_descriptors = collator._element_order_descriptors

        assert element_order_descriptors == expected_value


class DescribeMarginalCollator(object):
    """Unit-test suite for `cr.cube.collator.MarginalCollator` object.

    MarginalCollator computes element ordering for sort-by-marginal order transforms.
    Margins it can sort by include weighted-N (margin), unweighted-N (base), and "All"
    (percent).
    """

    def it_provides_an_interface_classmethod(self, request):
        dimension_ = instance_mock(request, Dimension)
        _init_ = initializer_mock(request, MarginalCollator)
        _display_order_ = property_mock(request, MarginalCollator, "_display_order")
        _display_order_.return_value = (-3, -1, -2, 1, 0, 2, 3)

        display_order = MarginalCollator.display_order(
            dimension_, ("vec", "tors"), ("inserted", "vectors")
        )

        _init_.assert_called_once_with(
            ANY, dimension_, ("vec", "tors"), ("inserted", "vectors")
        )
        assert display_order == (-3, -1, -2, 1, 0, 2, 3)

    def it_gathers_the_base_vector_marginal_values_to_help(
        self, request, _marginal_propname_prop_
    ):
        _marginal_propname_prop_.return_value = "margin"
        vectors_ = tuple(
            instance_mock(request, _CategoricalVector, margin=margin)
            for margin in (1.2, 3.1, 2.5, 1.7, 8.2)
        )
        collator = MarginalCollator(None, vectors_, None)

        element_values = collator._element_values

        assert element_values == (1.2, 3.1, 2.5, 1.7, 8.2)

    @pytest.mark.parametrize(
        "marginal_keyword, expected_value",
        (("unweighted_N", "base"), ("weighted_N", "margin")),
    )
    def it_maps_the_marginal_keyword_to_its_vector_property_name_to_help(
        self, request, marginal_keyword, expected_value
    ):
        property_mock(
            request,
            MarginalCollator,
            "_order_dict",
            return_value={"marginal": marginal_keyword},
        )
        collator = MarginalCollator(None, None, None)

        marginal_propname = collator._marginal_propname

        assert marginal_propname == expected_value

    def it_gathers_the_subtotal_marginal_values_to_help(
        self, request, _marginal_propname_prop_
    ):
        _marginal_propname_prop_.return_value = "base"
        subtotal_vectors_ = tuple(
            instance_mock(request, _BaseInsertedVector, base=1.0 * n)
            for n in range(1, 4)
        )
        collator = MarginalCollator(None, None, subtotal_vectors_)

        subtotal_values = collator._subtotal_values

        assert subtotal_values == (1.0, 2.0, 3.0)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _marginal_propname_prop_(self, request):
        return property_mock(request, MarginalCollator, "_marginal_propname")


class DescribeOpposingElementCollator(object):
    """Unit-test suite for `cr.cube.collator.OpposingElementCollator` object.

    OpposingElementCollator computes element ordering for order transforms of type
    "opposing_element". This would be like "order rows in descending order of the
    count in their 'Strongly Agree' column cell.
    """

    def it_provides_an_interface_classmethod(self, request):
        _init_ = initializer_mock(request, OpposingElementCollator)
        dimension_ = instance_mock(request, Dimension)
        _display_order_ = property_mock(
            request, OpposingElementCollator, "_display_order"
        )
        _display_order_.return_value = (-2, -1, -3, 1, 0, 3, 2)

        display_order = OpposingElementCollator.display_order(
            dimension_, ("opposing", "vectors")
        )

        _init_.assert_called_once_with(ANY, dimension_, ("opposing", "vectors"))
        assert display_order == (-2, -1, -3, 1, 0, 3, 2)

    def it_gathers_the_base_vector_measure_values_to_help(
        self, request, _measure_propname_prop_
    ):
        _measure_propname_prop_.return_value = "counts"
        property_mock(
            request,
            OpposingElementCollator,
            "_opposing_vector",
            return_value=instance_mock(
                request, _CategoricalVector, counts=np.array([2.5, 1.5, 1.0, 3.25])
            ),
        )
        collator = OpposingElementCollator(None, None)

        element_values = collator._element_values

        assert element_values == (2.5, 1.5, 1.0, 3.25)

    def it_finds_the_sort_key_opposing_vector_to_help(self, request, _order_dict_prop_):
        _order_dict_prop_.return_value = {"element_id": 2}
        opposing_vectors_ = tuple(
            instance_mock(request, _CategoricalVector, element_id=element_id)
            for element_id in (1, 2, 3)
        )
        collator = OpposingElementCollator(None, opposing_vectors_)

        opposing_vector = collator._opposing_vector

        assert opposing_vector is opposing_vectors_[1]

    def but_it_raises_ValueError_when_key_opposing_vector_is_not_present(
        self, request, _order_dict_prop_
    ):
        _order_dict_prop_.return_value = {"element_id": 666}
        opposing_vectors_ = tuple(
            instance_mock(request, _CategoricalVector, element_id=element_id)
            for element_id in (1, 2, 3)
        )
        collator = OpposingElementCollator(None, opposing_vectors_)

        with pytest.raises(ValueError):
            collator._opposing_vector

    def it_gathers_the_subtotal_values_to_help(self, request, _measure_propname_prop_):
        property_mock(
            request,
            OpposingElementCollator,
            "_subtotals",
            return_value=tuple(
                instance_mock(request, _Subtotal, addend_idxs=addend_idxs)
                for addend_idxs in ((0, 1), (4, 2, 3))
            ),
        )
        property_mock(
            request, OpposingElementCollator, "_measure_propname", return_value="counts"
        )
        property_mock(
            request,
            OpposingElementCollator,
            "_opposing_vector",
            return_value=instance_mock(
                request, _CategoricalVector, counts=np.array([1.0, 3.25, 2.5, 1.5, 8.0])
            ),
        )
        collator = OpposingElementCollator(None, None)

        subtotal_values = collator._subtotal_values

        assert subtotal_values == (4.25, 12.0)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _measure_propname_prop_(self, request):
        return property_mock(request, OpposingElementCollator, "_measure_propname")

    @pytest.fixture
    def _order_dict_prop_(self, request):
        return property_mock(request, OpposingElementCollator, "_order_dict")


class DescribeOpposingSubtotalCollator(object):
    """Unit-test suite for `cr.cube.collator.OpposingSubtotalCollator` object.

    OpposingSubtotalCollator computes element ordering for order transforms of type
    "opposing_subtotal". This would be like "order rows in descending order of the
    count in their 'Top 3' column cell.
    """

    def it_provides_an_interface_classmethod(self, request):
        _init_ = initializer_mock(request, OpposingSubtotalCollator)
        dimension_ = instance_mock(request, Dimension)
        _display_order_ = property_mock(
            request, OpposingSubtotalCollator, "_display_order"
        )
        _display_order_.return_value = (-3, -1, -2, 3, 0, 2, 1)

        display_order = OpposingSubtotalCollator.display_order(
            dimension_, ("opposing", "inserted", "vectors")
        )

        _init_.assert_called_once_with(
            ANY, dimension_, ("opposing", "inserted", "vectors")
        )
        assert display_order == (-3, -1, -2, 3, 0, 2, 1)

    def it_gathers_the_subtotal_values_to_help(self, request, _measure_propname_prop_):
        property_mock(
            request,
            OpposingSubtotalCollator,
            "_subtotals",
            return_value=tuple(
                instance_mock(request, _Subtotal, addend_idxs=addend_idxs)
                for addend_idxs in ((0, 1), (4, 2, 3))
            ),
        )
        property_mock(
            request,
            OpposingSubtotalCollator,
            "_measure_propname",
            return_value="counts",
        )
        property_mock(
            request,
            OpposingSubtotalCollator,
            "_opposing_subtotal",
            return_value=instance_mock(
                request, _BaseInsertedVector, counts=np.array([1.0, 3.5, 2.5, 1.5, 8.0])
            ),
        )
        collator = OpposingSubtotalCollator(None, None)

        subtotal_values = collator._subtotal_values

        assert subtotal_values == (4.5, 12.0)

    def it_finds_the_sort_key_opposing_subtotal_to_help(
        self, request, _order_dict_prop_
    ):
        _order_dict_prop_.return_value = {"insertion_id": 2}
        opposing_subtotals_ = tuple(
            instance_mock(request, _BaseInsertedVector, insertion_id=insertion_id)
            for insertion_id in (1, 2, 3)
        )
        collator = OpposingSubtotalCollator(None, opposing_subtotals_)

        opposing_subtotal = collator._opposing_subtotal

        assert opposing_subtotal is opposing_subtotals_[1]

    def but_it_raises_ValueError_when_key_opposing_subtotal_is_not_present(
        self, request, _order_dict_prop_
    ):
        _order_dict_prop_.return_value = {"insertion_id": 666}
        opposing_subtotals_ = tuple(
            instance_mock(request, _BaseInsertedVector, insertion_id=insertion_id)
            for insertion_id in (1, 2, 3)
        )
        collator = OpposingSubtotalCollator(None, opposing_subtotals_)

        with pytest.raises(ValueError):
            collator._opposing_subtotal

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _measure_propname_prop_(self, request):
        return property_mock(request, OpposingSubtotalCollator, "_measure_propname")

    @pytest.fixture
    def _order_dict_prop_(self, request):
        return property_mock(request, OpposingSubtotalCollator, "_order_dict")


class DescribePayloadOrderCollator(object):
    """Unit-test suite for `cr.cube.collator.PayloadOrderCollator` object."""

    @pytest.mark.parametrize(
        "element_ids, expected_value",
        (
            # --- no (valid) elements in dimension (not an expected case) ---
            ((), ()),
            # --- 1 element ---
            ((302,), ((0, 0, 302),)),
            # --- 3 elements in dimension ---
            ((47, 103, 18), ((0, 0, 47), (1, 1, 103), (2, 2, 18))),
        ),
    )
    def it_computes_the_element_order_descriptors_to_help(
        self, request, element_ids, expected_value
    ):
        property_mock(
            request, PayloadOrderCollator, "_element_ids", return_value=element_ids
        )
        collator = PayloadOrderCollator(None)

        element_order_descriptors = collator._element_order_descriptors

        assert element_order_descriptors == expected_value
