# encoding: utf-8

"""Unit test suite for `cr.cube.matrix` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _Element, _Subtotal
from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.matrix import (
    _AssembledVector,
    _BaseBaseMatrix,
    _BaseMatrixInsertedVector,
    _BaseTransformationVector,
    _BaseVector,
    _InsertedColumn,
    _InsertedRow,
    _OrderedVector,
    TransformedMatrix,
)

from ..unitutil import (
    ANY,
    call,
    class_mock,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class DescribeTransformedMatrix(object):
    """Unit test suite for `cr.cube.matrix.TransformedMatrix` object."""

    def it_provides_a_constructor_classmethod(self, request, base_matrix_, cube_):
        _BaseBaseMatrix_ = class_mock(request, "cr.cube.matrix._BaseBaseMatrix")
        _BaseBaseMatrix_.factory.return_value = base_matrix_
        dimensions_ = tuple(instance_mock(request, Dimension) for _ in range(2))
        _init_ = initializer_mock(request, TransformedMatrix)

        matrix = TransformedMatrix.matrix(cube_, dimensions_, slice_idx=7)

        _BaseBaseMatrix_.factory.assert_called_once_with(cube_, dimensions_, 7)
        _init_.assert_called_once_with(ANY, base_matrix_)
        assert isinstance(matrix, TransformedMatrix)

    def it_assembles_inserted_columns_into_base_columns_to_help(
        self,
        _base_columns_prop_,
        _inserted_columns_prop_,
        _inserted_rows_prop_,
        _assembled_vectors_,
    ):
        _base_columns_prop_.return_value = ("base", "columns")
        _inserted_columns_prop_.return_value = ("inserted", "columns")
        _inserted_rows_prop_.return_value = ("inserted", "rows")
        _assembled_vectors_.return_value = ("assembled", "columns")
        matrix = TransformedMatrix(None)

        assembled_columns = matrix._assembled_columns

        _assembled_vectors_.assert_called_once_with(
            matrix, ("base", "columns"), ("inserted", "columns"), ("inserted", "rows")
        )
        assert assembled_columns == ("assembled", "columns")

    def it_assembles_inserted_rows_into_base_rows_to_help(
        self,
        _base_rows_prop_,
        _inserted_rows_prop_,
        _inserted_columns_prop_,
        _assembled_vectors_,
    ):
        _base_rows_prop_.return_value = ("base", "rows")
        _inserted_rows_prop_.return_value = ("inserted", "rows")
        _inserted_columns_prop_.return_value = ("inserted", "columns")
        _assembled_vectors_.return_value = ("assembled", "rows")
        matrix = TransformedMatrix(None)

        assembled_rows = matrix._assembled_rows

        _assembled_vectors_.assert_called_once_with(
            matrix, ("base", "rows"), ("inserted", "rows"), ("inserted", "columns")
        )
        assert assembled_rows == ("assembled", "rows")

    @pytest.mark.parametrize(
        ("InsertedVectorCls", "OpposingInsertedVectorCls"),
        ((_InsertedColumn, _InsertedRow), (_InsertedRow, _InsertedColumn)),
    )
    def it_assembles_base_vectors_with_inserted_vectors_to_help(
        self, request, InsertedVectorCls, OpposingInsertedVectorCls, _AssembledVector_
    ):
        # --- base vectors ---
        top_base_vector_ = instance_mock(request, _OrderedVector, name="top")
        top_base_vector_.ordering = (0, 0, top_base_vector_)
        bot_base_vector_ = instance_mock(request, _OrderedVector, name="bot")
        bot_base_vector_.ordering = (1, 1, bot_base_vector_)
        base_vectors_ = (top_base_vector_, bot_base_vector_)
        # --- inserted vectors ---
        top_inserted_vector_ = instance_mock(request, InsertedVectorCls, name="top")
        top_inserted_vector_.ordering = (0, -2, top_inserted_vector_)
        mid_inserted_vector_ = instance_mock(request, InsertedVectorCls, name="mid")
        mid_inserted_vector_.ordering = (1, -1, mid_inserted_vector_)
        bot_inserted_vector_ = instance_mock(request, InsertedVectorCls, name="bot")
        bot_inserted_vector_.ordering = (sys.maxsize, -3, bot_inserted_vector_)
        inserted_vectors_ = (
            bot_inserted_vector_,
            top_inserted_vector_,
            mid_inserted_vector_,
        )
        # --- opposing inserted vectors ---
        opposing_inserted_vectors_ = tuple(
            instance_mock(request, OpposingInsertedVectorCls) for _ in range(3)
        )
        # --- _AssembledVector behaviors ---
        assembled_vectors_ = tuple(
            instance_mock(request, _AssembledVector) for _ in range(8)
        )
        _AssembledVector_.side_effect = iter(assembled_vectors_)
        matrix = TransformedMatrix(None)

        assembled_vectors = matrix._assembled_vectors(
            base_vectors_, inserted_vectors_, opposing_inserted_vectors_
        )

        assert _AssembledVector_.call_args_list == [
            call(top_inserted_vector_, opposing_inserted_vectors_, 0),
            call(top_base_vector_, opposing_inserted_vectors_, 0),
            call(mid_inserted_vector_, opposing_inserted_vectors_, 0),
            call(bot_base_vector_, opposing_inserted_vectors_, 1),
            call(bot_inserted_vector_, opposing_inserted_vectors_, 0),
        ]
        assert assembled_vectors == assembled_vectors_[:5]

    def it_constructs_its_inserted_columns_to_help(
        self,
        request,
        _columns_dimension_prop_,
        dimension_,
        _base_rows_prop_,
        _base_columns_prop_,
        unordered_matrix_,
        _InsertedColumn_,
    ):
        subtotals_ = tuple(
            instance_mock(request, _Subtotal, name="subtot[%i]" % i) for i in range(3)
        )
        inserted_columns_ = tuple(
            instance_mock(request, _InsertedColumn) for _ in range(5)
        )
        dimension_.dimension_type = DT.CAT
        dimension_.subtotals = subtotals_
        _columns_dimension_prop_.return_value = dimension_
        _base_rows_prop_.return_value = ("base", "rows")
        _base_columns_prop_.return_value = ("base", "columns")
        unordered_matrix_.table_margin = 73
        _InsertedColumn_.side_effect = iter(inserted_columns_)
        matrix = TransformedMatrix(unordered_matrix_)

        inserted_columns = matrix._inserted_columns

        assert _InsertedColumn_.call_args_list == [
            call(subtotals_[0], -3, 73, ("base", "rows"), ("base", "columns")),
            call(subtotals_[1], -2, 73, ("base", "rows"), ("base", "columns")),
            call(subtotals_[2], -1, 73, ("base", "rows"), ("base", "columns")),
        ]
        assert inserted_columns == inserted_columns_[:3]

    @pytest.mark.parametrize("dimension_type", (DT.CA, DT.MR))
    def but_it_constructs_no_inserted_columns_for_an_array_dimension(
        self, dimension_type, _columns_dimension_prop_, dimension_
    ):
        dimension_.dimension_type = dimension_type
        _columns_dimension_prop_.return_value = dimension_
        matrix = TransformedMatrix(None)

        inserted_columns = matrix._inserted_columns

        assert inserted_columns == ()

    def it_constructs_its_inserted_rows_to_help(
        self,
        request,
        _rows_dimension_prop_,
        dimension_,
        _base_columns_prop_,
        _base_rows_prop_,
        unordered_matrix_,
        _InsertedRow_,
    ):
        subtotals_ = tuple(
            instance_mock(request, _Subtotal, name="subtot[%i]" % i) for i in range(3)
        )
        inserted_rows_ = tuple(instance_mock(request, _InsertedRow) for _ in range(5))
        dimension_.dimension_type = DT.CAT
        dimension_.subtotals = subtotals_
        _rows_dimension_prop_.return_value = dimension_
        _base_columns_prop_.return_value = ("base", "columns")
        _base_rows_prop_.return_value = ("base", "rows")
        unordered_matrix_.table_margin = 302
        _InsertedRow_.side_effect = iter(inserted_rows_)
        matrix = TransformedMatrix(unordered_matrix_)

        inserted_rows = matrix._inserted_rows

        assert _InsertedRow_.call_args_list == [
            call(subtotals_[0], -3, 302, ("base", "rows"), ("base", "columns")),
            call(subtotals_[1], -2, 302, ("base", "rows"), ("base", "columns")),
            call(subtotals_[2], -1, 302, ("base", "rows"), ("base", "columns")),
        ]
        assert inserted_rows == inserted_rows_[:3]

    @pytest.mark.parametrize("dimension_type", (DT.CA, DT.MR))
    def but_it_constructs_no_inserted_rows_for_an_array_dimension(
        self, dimension_type, _rows_dimension_prop_, dimension_
    ):
        dimension_.dimension_type = dimension_type
        _rows_dimension_prop_.return_value = dimension_
        matrix = TransformedMatrix(None)

        inserted_rows = matrix._inserted_rows

        assert inserted_rows == ()

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _AssembledVector_(self, request):
        return class_mock(request, "cr.cube.matrix._AssembledVector")

    @pytest.fixture
    def _assembled_vectors_(self, request):
        return method_mock(request, TransformedMatrix, "_assembled_vectors")

    @pytest.fixture
    def _base_columns_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_base_columns")

    @pytest.fixture
    def base_matrix_(self, request):
        return instance_mock(request, _BaseBaseMatrix)

    @pytest.fixture
    def _base_rows_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_base_rows")

    @pytest.fixture
    def _columns_dimension_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_columns_dimension")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _InsertedColumn_(self, request):
        return class_mock(request, "cr.cube.matrix._InsertedColumn")

    @pytest.fixture
    def _inserted_columns_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_inserted_columns")

    @pytest.fixture
    def _InsertedRow_(self, request):
        return class_mock(request, "cr.cube.matrix._InsertedRow")

    @pytest.fixture
    def _inserted_rows_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_inserted_rows")

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_rows_dimension")

    @pytest.fixture
    def unordered_matrix_(self, request):
        return instance_mock(request, _BaseBaseMatrix)


class Describe_BaseMatrixInsertedVector(object):
    """Unit test suite for `cr.cube.matrix._BaseMatrixInsertedVector` object."""

    def it_knows_the_indices_of_its_addends(
        self, request, subtotal_, _base_vectors_prop_
    ):
        subtotal_.addend_ids = (2, 3)
        _base_vectors_prop_.return_value = tuple(
            instance_mock(request, _BaseVector, element_id=i + 1) for i in range(6)
        )
        inserted_vector = _BaseMatrixInsertedVector(subtotal_, None, None, None, None)

        addend_idxs = inserted_vector.addend_idxs

        np.testing.assert_equal(addend_idxs, (1, 2))

    @pytest.mark.parametrize("anchor_value", ("top", "bottom", 42))
    def it_knows_its_anchor(self, subtotal_, anchor_value):
        subtotal_.anchor = anchor_value
        inserted_vector = _BaseMatrixInsertedVector(subtotal_, None, None, None, None)

        anchor = inserted_vector.anchor

        assert anchor == anchor_value

    def it_knows_it_is_inserted(self):
        assert (
            _BaseMatrixInsertedVector(None, None, None, None, None).is_inserted is True
        )

    def it_knows_its_ordering_value(self, request):
        property_mock(request, _BaseMatrixInsertedVector, "_anchor_n", return_value=8)
        inserted_vector = _BaseMatrixInsertedVector(None, -4, None, None, None)

        ordering = inserted_vector.ordering

        assert ordering == (8, -4, inserted_vector)

    @pytest.mark.parametrize(
        ("anchor", "expected_value"), (("top", 0), ("bottom", sys.maxsize), (3, 3))
    )
    def it_knows_its_anchor_location(
        self, request, anchor, expected_value, _base_vectors_prop_
    ):
        _base_vectors_prop_.return_value = tuple(
            instance_mock(request, _BaseVector, name="base[%d]" % i, element_id=i + 1)
            for i in range(5)
        )
        property_mock(request, _BaseMatrixInsertedVector, "anchor", return_value=anchor)
        inserted_vector = _BaseMatrixInsertedVector(None, None, None, None, None)

        anchor_n = inserted_vector._anchor_n

        assert anchor_n == expected_value

    @pytest.mark.parametrize("addend_idxs", ((), (1,), (2, 3)))
    def it_gathers_its_addend_vectors_to_help(
        self, request, addend_idxs, _base_vectors_prop_
    ):
        property_mock(
            request, _BaseMatrixInsertedVector, "addend_idxs", return_value=addend_idxs
        )
        base_vectors_ = tuple(
            instance_mock(request, _BaseVector, name="base[i]") for i in range(6)
        )
        _base_vectors_prop_.return_value = base_vectors_
        inserted_vector = _BaseMatrixInsertedVector(None, -4, None, None, None)

        addend_vectors = inserted_vector._addend_vectors

        assert addend_vectors == tuple(
            vector for i, vector in enumerate(base_vectors_) if i in addend_idxs
        )

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_vectors_prop_(self, request):
        return property_mock(request, _BaseMatrixInsertedVector, "_base_vectors")

    @pytest.fixture
    def subtotal_(self, request):
        return instance_mock(request, _Subtotal)


class Describe_BaseTransformationVector(object):
    """Unit test suite for `cr.cube.matrix._BaseTransformationVector` object."""

    def it_knows_its_element_id(self, base_vector_):
        base_vector_.element_id = 42
        transformation_vector = _BaseTransformationVector(base_vector_)

        element_id = transformation_vector.element_id

        assert element_id == 42

    # fixture components ---------------------------------------------

    @pytest.fixture
    def base_vector_(self, request):
        return instance_mock(request, _BaseVector)


class Describe_BaseVector(object):
    """Unit test suite for `cr.cube.matrix._BaseVector` object."""

    def it_knows_its_element_id(self, element_):
        element_.element_id = 42
        base_vector = _BaseVector(element_, None)

        element_id = base_vector.element_id

        assert element_id == 42

    # fixture components ---------------------------------------------

    @pytest.fixture
    def element_(self, request):
        return instance_mock(request, _Element)
