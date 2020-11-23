# encoding: utf-8

"""Unit test suite for `cr.cube.matrix` module."""

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.matrix import Assembler, _BaseCubeResultMatrix

from ..unitutil import class_mock, instance_mock, method_mock, property_mock


class DescribeAssembler(object):
    """Unit test suite for `cr.cube.matrix.Assembler` object."""

    def it_knows_the_unweighted_counts(
        self,
        _cube_result_matrix_prop_,
        cube_result_matrix_,
        _SumSubtotals_,
        _assemble_matrix_,
    ):
        _cube_result_matrix_prop_.return_value = cube_result_matrix_
        _SumSubtotals_.blocks.return_value = [["A", "B"], ["C", "D"]]
        _assemble_matrix_.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assembler = Assembler(None, None, None)

        unweighted_counts = assembler.unweighted_counts

        _SumSubtotals_.blocks.assert_called_once_with(
            cube_result_matrix_, "unweighted_counts"
        )
        _assemble_matrix_.assert_called_once_with(assembler, [["A", "B"], ["C", "D"]])
        assert unweighted_counts == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def it_constructs_the_cube_result_matrix_to_help(
        self, request, cube_, dimension_, cube_result_matrix_
    ):
        _BaseCubeResultMatrix_ = class_mock(
            request, "cr.cube.matrix._BaseCubeResultMatrix"
        )
        _BaseCubeResultMatrix_.factory.return_value = cube_result_matrix_
        assembler = Assembler(cube_, (dimension_, dimension_), slice_idx=42)

        cube_result_matrix = assembler._cube_result_matrix

        _BaseCubeResultMatrix_.factory.assert_called_once_with(
            cube_, (dimension_, dimension_), 42
        )
        assert cube_result_matrix is cube_result_matrix_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _assemble_matrix_(self, request):
        return method_mock(request, Assembler, "_assemble_matrix")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def cube_result_matrix_(self, request):
        return instance_mock(request, _BaseCubeResultMatrix)

    @pytest.fixture
    def _cube_result_matrix_prop_(self, request):
        return property_mock(request, Assembler, "_cube_result_matrix")

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _SumSubtotals_(self, request):
        return class_mock(request, "cr.cube.matrix._SumSubtotals")


class Describe_BaseCubeResultMatrix(object):
    """Unit test suite for `cr.cube.matrix._BaseCubeResultMatrix` object."""

    @pytest.mark.parametrize(
        ("is_mr_aug", "has_means", "factory_method_name"),
        (
            (True, True, "_mr_aug_matrix_factory"),
            (True, False, "_mr_aug_matrix_factory"),
            (False, True, "_means_matrix_factory"),
            (False, False, "_regular_matrix_factory"),
        ),
    )
    def it_calls_the_correct_factory_method_for_appropriate_matrix_type(
        self, request, cube_, dimension_, is_mr_aug, has_means, factory_method_name
    ):
        cube_.is_mr_aug = is_mr_aug
        cube_.has_means = has_means
        cube_result_matrix_ = instance_mock(request, _BaseCubeResultMatrix)
        factory_method = method_mock(
            request,
            _BaseCubeResultMatrix,
            factory_method_name,
            return_value=cube_result_matrix_,
        )

        cube_result_matrix = _BaseCubeResultMatrix.factory(
            cube_, (dimension_, dimension_), slice_idx=71
        )

        factory_method.assert_called_once_with(cube_, (dimension_, dimension_), 71)
        assert cube_result_matrix is cube_result_matrix_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)
