# encoding: utf-8

"""Unit test suite for `cr.cube.matrix` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.matrix import (
    _AssembledVector,
    _BaseBaseMatrix,
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
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def _inserted_columns_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_inserted_columns")

    @pytest.fixture
    def _inserted_rows_prop_(self, request):
        return property_mock(request, TransformedMatrix, "_inserted_rows")
