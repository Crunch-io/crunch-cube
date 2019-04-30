# encoding: utf-8

import numpy as np
from mock import Mock
import pytest

from cr.cube.slices import (
    _AssembledVector,
    _CatXCatSlice,
    _CategoricalVector,
    _MultipleResponseVector,
    Insertions,
    InsertionRow,
    InsertionColumn,
    Assembler,
    Calculator,
    OrderedVector,
    OrderedSlice,
    OrderTransform,
)
from cr.cube.dimension import Dimension, _Subtotal, _Category
from ..unitutil import instance_mock, property_mock


class Describe_CatXCatSlice(object):
    def it_sets_raw_counts(self):
        counts = Mock()
        slice_ = _CatXCatSlice(counts)
        assert slice_._raw_counts == counts

    def it_provides_access_to_rows_and_columns(self):
        counts = np.arange(12).reshape(3, 4)
        slice_ = _CatXCatSlice(counts)

        # Check rows
        assert isinstance(slice_.rows, tuple)
        assert len(slice_.rows) == 3
        np.testing.assert_array_equal(
            slice_.rows[0]._raw_counts, np.array([0, 1, 2, 3])
        )
        np.testing.assert_array_equal(
            slice_.rows[1]._raw_counts, np.array([4, 5, 6, 7])
        )
        np.testing.assert_array_equal(
            slice_.rows[2]._raw_counts, np.array([8, 9, 10, 11])
        )

        # Check columns
        assert isinstance(slice_.columns, tuple)
        assert len(slice_.columns) == 4
        np.testing.assert_array_equal(
            slice_.columns[0]._raw_counts, np.array([0, 4, 8])
        )
        np.testing.assert_array_equal(
            slice_.columns[1]._raw_counts, np.array([1, 5, 9])
        )
        np.testing.assert_array_equal(
            slice_.columns[2]._raw_counts, np.array([2, 6, 10])
        )
        np.testing.assert_array_equal(
            slice_.columns[3]._raw_counts, np.array([3, 7, 11])
        )


class Describe_CategoricalVector(object):
    def it_sets_raw_counts(self):
        counts = Mock()
        row = _CategoricalVector(counts)
        assert row._raw_counts == counts

    def it_provides_values(self):
        counts = np.array([1, 2, 3])
        row = _CategoricalVector(counts)
        np.testing.assert_array_equal(row.values, counts)

    def it_calculates_margin(self):
        counts = np.array([1, 2, 3])
        row = _CategoricalVector(counts)
        assert row.margin == 6

    def it_calculates_proportions(self):
        counts = np.array([1, 2, 3])
        row = _CategoricalVector(counts)
        np.testing.assert_almost_equal(
            row.proportions, np.array([0.1666667, 0.3333333, 0.5])
        )


class Describe_MultipleResponseVector(object):
    def it_provides_values(self):
        counts = np.array([[1, 2, 3], [4, 5, 6]])
        row = _MultipleResponseVector(counts)
        np.testing.assert_array_equal(row._selected, np.array([1, 2, 3]))
        np.testing.assert_array_equal(row.values, np.array([1, 2, 3]))
        np.testing.assert_array_equal(row._not_selected, np.array([4, 5, 6]))

    def it_calculates_margin(self):
        """Margin needs to be a vector of selected = not-selected values."""
        counts = np.array([[1, 2, 3], [4, 5, 6]])
        row = _MultipleResponseVector(counts)
        np.testing.assert_array_equal(row.margin, np.array([5, 7, 9]))


class DescribeInsertions(object):
    def it_sets_dimensions_and_slice(self):
        dimensions, slice_ = Mock(), Mock()
        insertions = Insertions(dimensions, slice_)
        assert insertions._dimensions, insertions._slice == (dimensions, slice_)

    def it_provides_access_to_rows(self, _subtotals_prop_, addend_idxs_prop_):
        slice_ = _CatXCatSlice(np.arange(12).reshape(4, 3))
        _subtotals_prop_.return_value = [_Subtotal(None, None)]
        addend_idxs_prop_.return_value = (1, 2)
        insertions = Insertions((Dimension(None, None), None), slice_)
        assert len(insertions._rows) == 1
        np.testing.assert_array_equal(insertions._rows[0].values, [9, 11, 13])

    def it_provides_access_to_columns(self, _subtotals_prop_, addend_idxs_prop_):
        slice_ = _CatXCatSlice(np.arange(12).reshape(4, 3))
        _subtotals_prop_.return_value = [_Subtotal(None, None)]
        addend_idxs_prop_.return_value = (1, 2)
        insertions = Insertions((None, Dimension(None, None)), slice_)
        assert len(insertions._columns) == 1
        np.testing.assert_array_equal(insertions._columns[0].values, [3, 9, 15, 21])

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
    def it_sets_slice_and_subtotal(self):
        slice_ = Mock()
        subtotal = Mock()
        insertion_row = InsertionRow(slice_, subtotal)
        assert insertion_row._slice == slice_
        assert insertion_row._subtotal == subtotal

    def it_provides_values(self, addend_idxs_prop_, values_fixture):
        slice_, subtotal_indexes, expected_row_counts = values_fixture
        addend_idxs_prop_.return_value = subtotal_indexes
        insertion_row = InsertionRow(slice_, _Subtotal(None, None))
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
    def it_provides_assembled_values(self, assembled_row_fixture):
        raw_counts, insertion_cols, expected = assembled_row_fixture
        row = _AssembledVector(raw_counts, insertion_cols)
        np.testing.assert_array_equal(row.values, expected)

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
                InsertionColumn,
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
                InsertionColumn,
                anchor=anchor,
                addend_idxs=np.array(addend_idxs),
            )
            for anchor, addend_idxs in insertion_cols_counts
        )
        return _CategoricalVector(np.array(raw_counts)), insertion_cols, expected


class DescribeAssembler(object):
    def it_sets_slice_and_insertions(self):
        slice_, insertions = Mock(), Mock()
        assembler = Assembler(slice_, insertions)
        assert assembler._slice, assembler._insertions == (slice_, insertions)

    def it_provides_rows(self, counts_fixture):
        slice_, insertions, expected = counts_fixture
        assembler = Assembler(slice_, insertions)
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
        return slice_, insertions, np.array(expected_counts)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, Dimension, "_subtotals")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension, _subtotals=tuple())


class DescribeCalculator(object):
    def it_provides_row_proportions(self, row_proportions_fixture):
        slice_, insertions, expected = row_proportions_fixture
        calc = Calculator(Assembler(slice_, insertions))
        row_proportions = calc.row_proportions
        np.testing.assert_almost_equal(row_proportions, expected)

    def it_provides_row_margin(self, row_margin_fixture):
        slice_, insertions, expected = row_margin_fixture
        calc = Calculator(Assembler(slice_, insertions))
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
        return slice_, insertions, np.array(expected_row_proportions)

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
        return slice_, insertions, np.array(expected_row_margin)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, Dimension, "_subtotals")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension, _subtotals=tuple())


class DescribeOrderTransform(object):
    def it_initiates_dimensions_and_ordered_ids(self):
        dimensions = Mock()
        ordered_ids = Mock()
        transform = OrderTransform(dimensions, ordered_ids)
        assert transform._dimensions == dimensions
        assert transform._ordered_ids == ordered_ids

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
    def it_initiates_slice_and_reordering(self):
        slice_ = Mock()
        order_transform = Mock()
        ordered_slice = OrderedSlice(slice_, order_transform)
        assert ordered_slice._slice == slice_
        assert ordered_slice._transform == order_transform

    def it_reodrders_rows(self, order_rows_fixture):
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
    def it_sets_order(self):
        order = Mock()
        base_vector = Mock()
        ordered_vector = OrderedVector(base_vector, order)
        assert ordered_vector._vector == base_vector
        assert ordered_vector._order == order

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
