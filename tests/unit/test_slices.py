# encoding: utf-8

"""Unit test suite for `cr.cube.slices` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from mock import Mock
import pytest

from cr.cube.slices import (
    _AssembledVector,
    _CatXCatSlice,
    _MrXCatSlice,
    _CategoricalVector,
    _MultipleResponseVector,
    _InsertionColumn,
    _InsertionRow,
    Insertions,
    Assembler,
    Calculator,
    OrderedVector,
    OrderedSlice,
    OrderTransform,
    Transforms,
    PrunedVector,
    PrunedSlice,
)
from cr.cube.dimension import Dimension, _Subtotal, _Category
from ..unitutil import instance_mock, property_mock


class Describe_CategoricalVector(object):
    """Unit-test suite for `cr.cube.slices._CategoricalVector` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_sets_raw_counts(self):
        counts, base_counts, label, margin = Mock(), Mock(), Mock(), Mock()
        row = _CategoricalVector(counts, base_counts, label, margin)
        assert row._counts == counts
        assert row._base_counts == base_counts
        assert row._label == label
        assert row._table_margin == margin

    def it_provides_values(self):
        counts = np.array([1, 2, 3])
        row = _CategoricalVector(counts, None, None, None)
        np.testing.assert_array_equal(row.values, counts)

    def it_calculates_margin(self):
        counts = np.array([1, 2, 3])
        row = _CategoricalVector(counts, None, None, None)
        assert row.margin == 6

    def it_calculates_proportions(self):
        counts = np.array([1, 2, 3])
        row = _CategoricalVector(counts, None, None, None)
        np.testing.assert_almost_equal(
            row.proportions, np.array([0.1666667, 0.3333333, 0.5])
        )


class Describe_MultipleResponseVector(object):
    """Unit-test suite for `cr.cube.slices._MultipleResponseVector` object."""

    def it_provides_values(self):
        counts = np.array([[1, 2, 3], [4, 5, 6]])
        row = _MultipleResponseVector(counts, None, None, None)
        np.testing.assert_array_equal(row._selected, np.array([1, 2, 3]))
        np.testing.assert_array_equal(row.values, np.array([1, 2, 3]))
        np.testing.assert_array_equal(row._not_selected, np.array([4, 5, 6]))

    def it_calculates_margin(self):
        """Margin needs to be a vector of selected = not-selected values."""
        counts = np.array([[1, 2, 3], [4, 5, 6]])
        row = _MultipleResponseVector(counts, None, None, None)
        np.testing.assert_array_equal(row.margin, np.array([5, 7, 9]))


class DescribeInsertions(object):
    """Unit-test suite for `cr.cube.slices.Insertions` object."""

    def it_sets_dimensions_and_slice(self):
        dimensions, slice_ = Mock(), Mock()
        insertions = Insertions(dimensions, slice_)
        assert insertions._dimensions, insertions._slice == (dimensions, slice_)

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_access_to_rows(self, _subtotals_prop_, addend_idxs_prop_):
        slice_ = _CatXCatSlice(np.arange(12).reshape(4, 3))
        _subtotals_prop_.return_value = [_Subtotal(None, None)]
        addend_idxs_prop_.return_value = (1, 2)
        insertions = Insertions((Dimension(None, None), None), slice_)
        assert len(insertions._rows) == 1
        np.testing.assert_array_equal(insertions._rows[0].values, [9, 11, 13])

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_access_to_columns(self, _subtotals_prop_, addend_idxs_prop_):
        slice_ = _CatXCatSlice(np.arange(12).reshape(4, 3))
        _subtotals_prop_.return_value = [_Subtotal(None, None)]
        addend_idxs_prop_.return_value = (1, 2)
        insertions = Insertions((None, Dimension(None, None)), slice_)
        assert len(insertions._columns) == 1
        np.testing.assert_array_equal(insertions._columns[0].values, [3, 9, 15, 21])

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_knows_its_intersections(self, request, intersections_fixture):
        slice_ = _CatXCatSlice(np.arange(12).reshape(4, 3))
        row_dimension, col_dimension, expected = intersections_fixture
        insertions = Insertions((row_dimension, col_dimension), slice_)
        intersections = insertions.intersections
        np.testing.assert_array_equal(intersections, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (((2, (2, 3)),), ((1, (0, 1)),), 32),
            (((0, (0, 1)),), ((0, (0, 1)),), 6),
            (((0, (0, 1)),), ((0, (0, 1)), (1, (0, 2))), [[6, 9]]),
            (((0, (0, 1)), (1, (2, 3))), ((0, (0, 1)),), [[6], [24]]),
            (
                ((0, (0, 1)), (1, (2, 3))),
                ((0, (0, 1)), (1, (0, 2))),
                [[6, 9], [24, 27]],
            ),
        ]
    )
    def intersections_fixture(self, request):
        row_subtotals, col_subtotals, expected = request.param
        row_dimension = instance_mock(
            request,
            Dimension,
            _subtotals=[
                instance_mock(
                    request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
                )
                for anchor_idx, addend_idxs in row_subtotals
            ],
        )
        col_dimension = instance_mock(
            request,
            Dimension,
            _subtotals=[
                instance_mock(
                    request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
                )
                for anchor_idx, addend_idxs in col_subtotals
            ],
        )
        return row_dimension, col_dimension, np.array(expected)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, Dimension, "_subtotals")

    @pytest.fixture
    def addend_idxs_prop_(self, request):
        return property_mock(request, _Subtotal, "addend_idxs")


class DescribeInsertionsRow(object):
    """Unit-test suite for `cr.cube.slices._InsertionRow` object."""

    def it_sets_slice_and_subtotal(self):
        slice_ = Mock()
        subtotal = Mock()
        insertion_row = _InsertionRow(slice_, subtotal)
        assert insertion_row._slice == slice_
        assert insertion_row._subtotal == subtotal

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_values(self, addend_idxs_prop_, values_fixture):
        slice_, subtotal_indexes, expected_row_counts = values_fixture
        addend_idxs_prop_.return_value = subtotal_indexes
        insertion_row = _InsertionRow(slice_, _Subtotal(None, None))
        np.testing.assert_array_equal(insertion_row.values, expected_row_counts)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([[1, 2, 3], [4, 5, 6]], (0,), [1, 2, 3]),
            ([[1, 2, 3], [4, 5, 6]], (1,), [4, 5, 6]),
            ([[1, 2, 3], [4, 5, 6]], (0, 1), [5, 7, 9]),
            ([[1, 2], [3, 4], [5, 6]], (0, 2), [6, 8]),
        ]
    )
    def values_fixture(self, request):
        counts, subtotal_indexes, expected_row_counts = request.param
        return _CatXCatSlice(counts), subtotal_indexes, expected_row_counts

    # fixture components ---------------------------------------------

    @pytest.fixture
    def addend_idxs_prop_(self, request):
        return property_mock(request, _Subtotal, "addend_idxs")


class Describe_AssembledVector(object):
    """Unit-test suite for `cr.cube.slices._AssembledVector` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_assembled_values(self, assembled_row_fixture):
        raw_counts, insertion_cols, expected = assembled_row_fixture
        row = _AssembledVector(raw_counts, insertion_cols)
        np.testing.assert_array_equal(row.values, expected)

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_assembled_proportions(self, assembled_row_proportions_fixture):
        raw_counts, insertion_cols, expected = assembled_row_proportions_fixture
        row = _AssembledVector(raw_counts, insertion_cols)
        np.testing.assert_array_equal(row.proportions, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([1, 2, 2], ((1, [0, 1]),), [0.2, 0.4, 0.6, 0.4]),
            ([1, 2, 2], ((2, [0, 1]),), [0.2, 0.4, 0.4, 0.6]),
        ]
    )
    def assembled_row_proportions_fixture(self, request):
        raw_counts, insertion_cols_counts, expected = request.param
        insertion_cols = tuple(
            instance_mock(
                request,
                _InsertionColumn,
                anchor=anchor,
                addend_idxs=np.array(addend_idxs),
            )
            for anchor, addend_idxs in insertion_cols_counts
        )
        return _CategoricalVector(np.array(raw_counts)), insertion_cols, expected

    @pytest.fixture(
        params=[
            ([1, 2, 3], ((1, [0, 1]),), [1, 2, 3, 3]),
            ([1, 2, 3], ((2, [0, 1]),), [1, 2, 3, 3]),
            ([1, 2, 3], ((0, [1, 2]),), [1, 5, 2, 3]),
            ([1, 2, 3], (("top", [1, 2]),), [5, 1, 2, 3]),
            ([1, 2, 3], (("bottom", [1, 2]),), [1, 2, 3, 5]),
            ([1, 2, 3], ((0, [0, 1]), ("bottom", [1, 2])), [1, 3, 2, 3, 5]),
            (
                [1, 2, 3],
                (("top", [2]), (0, [0, 1]), ("bottom", [1, 2])),
                [3, 1, 3, 2, 3, 5],
            ),
        ]
    )
    def assembled_row_fixture(self, request):
        raw_counts, insertion_cols_counts, expected = request.param
        insertion_cols = tuple(
            instance_mock(
                request,
                _InsertionColumn,
                anchor=anchor,
                addend_idxs=np.array(addend_idxs),
            )
            for anchor, addend_idxs in insertion_cols_counts
        )
        return _CategoricalVector(np.array(raw_counts)), insertion_cols, expected


class DescribeAssembler(object):
    """Unit-test suite for `cr.cube.slices.Assembler` object."""

    def it_sets_slice_and_insertions(self):
        slice_, insertions = Mock(), Mock()
        assembler = Assembler(slice_, insertions)
        assert assembler._slice, assembler._insertions == (slice_, insertions)

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_rows(self, counts_fixture):
        slice_, transforms, expected = counts_fixture
        assembler = Assembler(slice_, transforms)
        actual_counts = np.array([row.values for row in assembler.rows])
        np.testing.assert_array_equal(actual_counts, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([[1, 2], [3, 4]], [], [[1, 2], [3, 4]]),
            ([[1, 2], [3, 4]], [(0, (0,))], [[1, 2], [1, 2], [3, 4]]),
            ([[1, 2], [3, 4]], [(1, (0,))], [[1, 2], [3, 4], [1, 2]]),
            (
                [[1, 2], [3, 4], [5, 6]],
                [("top", (1,)), (2, (1, 2))],
                [[3, 4], [1, 2], [3, 4], [5, 6], [8, 10]],
            ),
        ]
    )
    def counts_fixture(self, request, _subtotals_prop_, dimension_):
        counts, row_subtotals, expected_counts = request.param
        slice_ = _CatXCatSlice(counts)
        dimensions = (Dimension(None, None), dimension_)
        _subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        insertions = Insertions(dimensions, slice_)
        transforms = Transforms(None, None, insertions)
        return slice_, transforms, np.array(expected_counts)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, Dimension, "_subtotals")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension, _subtotals=tuple())


class DescribeCalculator(object):
    """Unit-test suite for `cr.cube.slices.Calculator` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_row_proportions(self, row_proportions_fixture):
        slice_, transforms, expected = row_proportions_fixture
        calc = Calculator(Assembler(slice_, transforms))
        row_proportions = calc.row_proportions
        np.testing.assert_almost_equal(row_proportions, expected)

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_row_margin(self, row_margin_fixture):
        slice_, transforms, expected = row_margin_fixture
        calc = Calculator(Assembler(slice_, transforms))
        margin = calc.row_margin
        np.testing.assert_almost_equal(margin, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (
                [[1, 2], [3, 4]],
                [],
                [[0.33333333, 0.66666667], [0.42857143, 0.57142857]],
            ),
            (
                [[1, 2], [3, 4]],
                [(0, (0,))],
                [
                    [0.33333333, 0.66666667],
                    [0.33333333, 0.66666667],
                    [0.42857143, 0.57142857],
                ],
            ),
            (
                [[1, 2], [3, 4]],
                [(1, (0,))],
                [
                    [0.33333333, 0.66666667],
                    [0.42857143, 0.57142857],
                    [0.33333333, 0.66666667],
                ],
            ),
            (
                [[1, 2], [3, 4], [5, 6]],
                [("top", (1,)), (2, (1, 2))],
                [
                    [0.42857143, 0.57142857],
                    [0.33333333, 0.66666667],
                    [0.42857143, 0.57142857],
                    [0.45454545, 0.54545455],
                    [0.44444444, 0.55555556],
                ],
            ),
        ]
    )
    def row_proportions_fixture(self, request, _subtotals_prop_, dimension_):
        counts, row_subtotals, expected_row_proportions = request.param
        slice_ = _CatXCatSlice(counts)
        dimensions = (Dimension(None, None), dimension_)
        _subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        insertions = Insertions(dimensions, slice_)
        transforms = Transforms(None, None, insertions)
        return slice_, transforms, np.array(expected_row_proportions)

    @pytest.fixture(
        params=[
            ([[1, 2], [3, 4]], [], [3, 7]),
            ([[1, 2], [3, 4]], [(0, (0,))], [3, 3, 7]),
            ([[1, 2], [3, 4]], [(1, (0,))], [3, 7, 3]),
            ([[1, 2], [3, 4], [5, 6]], [("top", (1,)), (2, (1, 2))], [7, 3, 7, 11, 18]),
        ]
    )
    def row_margin_fixture(self, request, _subtotals_prop_, dimension_):
        counts, row_subtotals, expected_row_margin = request.param
        slice_ = _CatXCatSlice(counts)
        dimensions = (Dimension(None, None), dimension_)
        _subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        insertions = Insertions(dimensions, slice_)
        transforms = Transforms(None, None, insertions)
        return slice_, transforms, np.array(expected_row_margin)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, Dimension, "_subtotals")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension, _subtotals=tuple())


class DescribeOrderTransform(object):
    """Unit-test suite for `cr.cube.slices.OrderTransform` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_initiates_dimensions_and_ordered_ids(self):
        dimensions = Mock()
        ordered_ids = Mock()
        transform = OrderTransform(dimensions, ordered_ids)
        assert transform._dimensions == dimensions
        assert transform._ordered_ids == ordered_ids

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_provides_rows_order(self, rows_order_fixture):
        row_dimension, ordered_element_ids, expected = rows_order_fixture
        dimensions = (row_dimension, None)
        transform = OrderTransform(dimensions, (ordered_element_ids, None))
        np.testing.assert_array_equal(transform.row_order, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([1, 2, 3], [1, 2, 3], [0, 1, 2]),
            ([1, 2, 3], [2, 1, 3], [1, 0, 2]),
            ([1, 2, 3], [3, 2, 1], [2, 1, 0]),
        ]
    )
    def rows_order_fixture(self, request):
        element_ids, ordered_element_ids, expected_order = request.param
        row_dimension = instance_mock(
            request,
            Dimension,
            valid_elements=[
                instance_mock(request, _Category, element_id=element_id)
                for element_id in element_ids
            ],
        )
        return row_dimension, ordered_element_ids, np.array(expected_order)


class DescribeOrderedSlice(object):
    """Unit-test suite for `cr.cube.slices.OrderedSlice` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_initiates_slice_and_reordering(self):
        slice_ = Mock()
        ordering = Mock()
        ordered_slice = OrderedSlice(slice_, ordering)
        assert ordered_slice._slice == slice_
        assert ordered_slice._ordering == ordering

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_reorders_rows(self, order_rows_fixture):
        ordered_slice, expected = order_rows_fixture
        for row, expected_row_values in zip(ordered_slice.rows, expected):
            assert row.values.tolist() == expected_row_values

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([1, 2, 3], [[1, 2], [3, 4], [5, 6]], [1, 2, 3], [[1, 2], [3, 4], [5, 6]]),
            ([1, 2, 3], [[1, 2], [3, 4], [5, 6]], [2, 1, 3], [[3, 4], [1, 2], [5, 6]]),
            ([1, 2, 3], [[1, 2], [3, 4], [5, 6]], [3, 2, 1], [[5, 6], [3, 4], [1, 2]]),
        ]
    )
    def order_rows_fixture(self, request):
        element_ids, counts, ordered_ids, expected = request.param
        row_dimension = instance_mock(
            request,
            Dimension,
            valid_elements=[
                instance_mock(request, _Category, element_id=element_id)
                for element_id in element_ids
            ],
        )
        transform = OrderTransform((row_dimension, None), (np.array(ordered_ids), None))
        return OrderedSlice(_CatXCatSlice(np.array(counts)), transform), expected


class DescribeOrderedVector(object):
    """Unit-test suite for `cr.cube.slices.OrderedVector` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_sets_order(self):
        order = Mock()
        base_vector = Mock()
        ordered_vector = OrderedVector(base_vector, order)
        assert ordered_vector._vector == base_vector
        assert ordered_vector._order == order

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_orders_values(self, order_values_fixture):
        vector, order, expected = order_values_fixture
        ordered_vector = OrderedVector(vector, order)
        np.testing.assert_array_equal(ordered_vector.values, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([1, 2, 3], [0, 1, 2], [1, 2, 3]),
            ([1, 2, 3], [1, 0, 2], [2, 1, 3]),
            ([1, 2, 3], [2, 1, 0], [3, 2, 1]),
        ]
    )
    def order_values_fixture(self, request):
        counts, order, expected = request.param
        return _CategoricalVector(np.array(counts)), np.array(order), expected


class DescribeTransforms(object):
    """Unit-test suite for `cr.cube.slices.Transforms` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_initiates_transforms(self):
        insertions = Mock()
        ordering = Mock()
        pruning = Mock()
        transforms = Transforms(ordering, pruning, insertions)
        assert transforms._ordering == ordering
        assert transforms._pruning == pruning
        assert transforms._insertions == insertions


class DescribePrunedSlice(object):
    """Unit-test suite for `cr.cube.slices.PrunedSlice` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_initiates_slice(self):
        slice_ = Mock()
        pruned_slice = PrunedSlice(slice_)
        assert pruned_slice._slice == slice_

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_prunes_rows_and_columns(self, pruned_slice_fixture):
        pruned_slice, expected = pruned_slice_fixture
        np.testing.assert_array_equal(
            np.array([row.values for row in pruned_slice.rows]), expected
        )
        np.testing.assert_array_equal(
            np.array([column.values for column in pruned_slice.columns]).T, expected
        )

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (_CatXCatSlice, [[1, 2], [3, 4]], [[1, 2], [3, 4]]),
            (_CatXCatSlice, [[1, 2], [0, 0], [3, 4]], [[1, 2], [3, 4]]),
            (_CatXCatSlice, [[1, 0, 1], [0, 0, 0], [3, 0, 4]], [[1, 1], [3, 4]]),
            (_MrXCatSlice, [[[1, 2], [0, 1]], [[3, 4], [1, 0]]], [[1, 2], [3, 4]]),
            (
                _MrXCatSlice,
                [[[1, 2, 0], [0, 1, 0]], [[3, 4, 0], [1, 0, 0]]],
                [[1, 2], [3, 4]],
            ),
            (
                _MrXCatSlice,
                [
                    [[1, 2, 0], [0, 1, 0]],
                    [[3, 4, 0], [1, 0, 0]],
                    [[0, 0, 0], [0, 0, 0]],
                ],
                [[1, 2], [3, 4]],
            ),
        ]
    )
    def pruned_slice_fixture(self, request):
        cls_, raw_counts, expected = request.param
        return PrunedSlice(cls_(np.array(raw_counts))), np.array(expected)


class DescribePrunedVector(object):
    """Unit-test suite for `cr.cube.slices.PrunedVector` object."""

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_initiates_base_vector_and_opposite_vectors(self):
        vector = Mock()
        opposite_vectors = Mock()
        pruned_vector = PrunedVector(vector, opposite_vectors)
        assert pruned_vector._vector == vector
        assert pruned_vector._opposite_vectors == opposite_vectors

    @pytest.mark.xfail(reason="FrozenSlice WIP", strict=True)
    def it_prunes_elements(self, pruned_elements_fixture):
        vector, opposite_vectors, expected = pruned_elements_fixture
        pruned_vector = PrunedVector(vector, opposite_vectors)
        np.testing.assert_array_equal(pruned_vector.values, expected)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (_CategoricalVector, [1, 2, 3], _CategoricalVector, [1, 2, 3], [1, 2, 3]),
            (_CategoricalVector, [1, 2, 3], _CategoricalVector, [0, 1, 2], [2, 3]),
            (
                _MultipleResponseVector,
                [[1, 2, 3], [4, 5, 6]],
                _CategoricalVector,
                [0, 1, 2],
                [2, 3],
            ),
            (
                _CategoricalVector,
                [1, 2, 3],
                _MultipleResponseVector,
                [[0, 0], [1, 2], [3, 4]],
                [2, 3],
            ),
            (
                _MultipleResponseVector,
                [[1, 2, 3], [4, 5, 6]],
                _MultipleResponseVector,
                [[3, 4], [0, 0], [3, 4]],
                [1, 3],
            ),
        ]
    )
    def pruned_elements_fixture(self, request):
        cls_, values, opposite_cls, margins, expected = request.param
        vector = cls_(np.array(values))
        opposite_vectors = [
            instance_mock(request, opposite_cls, margin=np.array(margin))
            for margin in margins
        ]
        return vector, opposite_vectors, expected
