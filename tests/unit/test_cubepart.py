# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from mock import Mock
import pytest

from cr.cube.cubepart import _Slice
from cr.cube.dimension import Dimension, _Subtotal
from cr.cube.matrix import (
    _AssembledVector,
    _CatXCatMatrix,
    _CategoricalVector,
    _InsertionColumn,
    _InsertionRow,
    _MatrixWithHidden,
    _MrXCatMatrix,
    _MultipleResponseVector,
    _OrderedMatrix,
    _OrderedVector,
    TransformedMatrix,
    _VectorAfterHiding,
)
from ..unitutil import instance_mock, property_mock


class DescribeIntegrated_Slice(object):
    """Partial-integration test suite for `cr.cube.cubepart._Slice` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_knows_the_row_proportions(self, row_proportions_fixture, _assembler_prop_):
        slice_, insertions, expected = row_proportions_fixture
        _assembler_prop_.return_value = TransformedMatrix(
            slice_, None, insertions, None
        )
        slice_ = _Slice(None, None, None, None, None)

        row_proportions = slice_.row_proportions

        np.testing.assert_almost_equal(row_proportions, expected)

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_knows_the_rows_margin(self, row_margin_fixture, _assembler_prop_):
        slice_, insertions, expected = row_margin_fixture
        _assembler_prop_.return_value = TransformedMatrix(
            slice_, None, insertions, None
        )
        slice_ = _Slice(None, None)

        margin = slice_.row_margin

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
    def row_proportions_fixture(self, request, subtotals_prop_, dimension_):
        counts, row_subtotals, expected_row_proportions = request.param
        dimensions = (Dimension(None, None), dimension_)
        slice_ = _CatXCatMatrix(dimensions, counts, None)
        subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        return slice_, np.array(expected_row_proportions)

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
        slice_ = _CatXCatMatrix(counts)
        _subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        return slice_, np.array(expected_row_margin)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _assembler_prop_(self, request):
        return property_mock(request, _Slice, "_assembler")

    @pytest.fixture
    def subtotals_prop_(self, request):
        return property_mock(request, Dimension, "subtotals")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension, subtotals=tuple())


class Describe_CategoricalVector(object):
    """Unit-test suite for `cr.cube.vector._CategoricalVector` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
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
    """Unit-test suite for `cr.cube.vector._MultipleResponseVector` object."""

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


class Describe_InsertionRow(object):
    """Unit-test suite for `cr.cube.vector._InsertionRow` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
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
        return _CatXCatMatrix(counts), subtotal_indexes, expected_row_counts

    # fixture components ---------------------------------------------

    @pytest.fixture
    def addend_idxs_prop_(self, request):
        return property_mock(request, _Subtotal, "addend_idxs")


class Describe_AssembledVector(object):
    """Unit-test suite for `cr.cube.vector._AssembledVector` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_provides_assembled_values(self, assembled_row_fixture):
        raw_counts, insertion_cols, expected = assembled_row_fixture
        row = _AssembledVector(raw_counts, insertion_cols)
        np.testing.assert_array_equal(row.values, expected)

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
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


class DescribeIntegratedTransformedMatrix(object):
    """Partial-integration test suite for `cr.cube.matrix.TransformedMatrix` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_provides_rows(self, counts_fixture):
        slice_, insertions, expected = counts_fixture
        assembler = TransformedMatrix(slice_, None, insertions, None)
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
        slice_ = _CatXCatMatrix(counts)
        _subtotals_prop_.return_value = [
            instance_mock(
                request, _Subtotal, anchor_idx=anchor_idx, addend_idxs=addend_idxs
            )
            for anchor_idx, addend_idxs in row_subtotals
        ]
        return slice_, np.array(expected_counts)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, Dimension, "_subtotals")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension, _subtotals=tuple())


class Describe_OrderedMatrix(object):
    """Unit-test suite for `cr.cube.matrix._OrderedMatrix` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
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
        # row_dimension = instance_mock(
        #     request,
        #     Dimension,
        #     valid_elements=[
        #         instance_mock(request, _Element, element_id=element_id)
        #         for element_id in element_ids
        #     ],
        # )
        # transform = _OrderTransform(
        #     (row_dimension, None), (np.array(ordered_ids), None)
        # )
        return _OrderedMatrix(_CatXCatMatrix(np.array(counts))), expected


class Describe_OrderedVector(object):
    """Unit-test suite for `cr.cube.vector._OrderedVector` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_sets_order(self):
        order = Mock()
        base_vector = Mock()
        ordered_vector = _OrderedVector(base_vector, order)
        assert ordered_vector._vector == base_vector
        assert ordered_vector._order == order

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_orders_values(self, order_values_fixture):
        vector, order, expected = order_values_fixture
        ordered_vector = _OrderedVector(vector, order)
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


class Describe_MatrixWithHidden(object):
    """Unit-test suite for `cr.cube.matrix._MatrixWithHidden` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_initiates_slice(self):
        slice_ = Mock()
        pruned_slice = _MatrixWithHidden(slice_)
        assert pruned_slice._slice == slice_

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
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
            (_CatXCatMatrix, [[1, 2], [3, 4]], [[1, 2], [3, 4]]),
            (_CatXCatMatrix, [[1, 2], [0, 0], [3, 4]], [[1, 2], [3, 4]]),
            (_CatXCatMatrix, [[1, 0, 1], [0, 0, 0], [3, 0, 4]], [[1, 1], [3, 4]]),
            (_MrXCatMatrix, [[[1, 2], [0, 1]], [[3, 4], [1, 0]]], [[1, 2], [3, 4]]),
            (
                _MrXCatMatrix,
                [[[1, 2, 0], [0, 1, 0]], [[3, 4, 0], [1, 0, 0]]],
                [[1, 2], [3, 4]],
            ),
            (
                _MrXCatMatrix,
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
        return _MatrixWithHidden(cls_(np.array(raw_counts))), np.array(expected)


class Describe_VectorAfterHiding(object):
    """Unit-test suite for `cr.cube.vector._VectorAfterHiding` object."""

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_initiates_base_vector_and_opposite_vectors(self):
        vector = Mock()
        opposite_vectors = Mock()
        pruned_vector = _VectorAfterHiding(vector, opposite_vectors)
        assert pruned_vector._vector == vector
        assert pruned_vector._opposite_vectors == opposite_vectors

    @pytest.mark.xfail(reason="_Slice WIP", strict=True)
    def it_prunes_elements(self, pruned_elements_fixture):
        vector, opposite_vectors, expected = pruned_elements_fixture
        pruned_vector = _VectorAfterHiding(vector, opposite_vectors)
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
