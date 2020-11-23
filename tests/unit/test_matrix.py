# encoding: utf-8

"""Unit test suite for `cr.cube.matrix` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.matrix import (
    Assembler,
    _BaseCubeResultMatrix,
    _CatXCatMatrix,
    _CatXMrMatrix,
    _MrXCatMatrix,
    _MrXMrMatrix,
)

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

    @pytest.mark.parametrize(
        ("slice_idx", "dim_types", "expected"),
        (
            # <= 2D
            (0, (DT.CAT,), np.s_[:]),
            (1, (DT.CAT,), np.s_[:]),
            (0, (DT.CAT, DT.CAT), np.s_[:]),
            (1, (DT.CAT, DT.CAT), np.s_[:]),
            # 3D, no MR as tabs
            (0, (DT.CAT, DT.CAT, DT.CAT), np.s_[0]),
            (1, (DT.CAT, DT.CAT, DT.CAT), np.s_[1]),
            (2, (DT.CAT, DT.CAT, DT.CAT), np.s_[2]),
            # 3D, MR as tabs
            (0, (DT.MR, DT.CAT, DT.CAT), np.s_[0, 0]),
            (1, (DT.MR, DT.CAT, DT.CAT), np.s_[1, 0]),
            (2, (DT.MR, DT.CAT, DT.CAT), np.s_[2, 0]),
        ),
    )
    def it_knows_its_regular_matrix_slice_for_counts_to_help(
        self, cube_, slice_idx, dim_types, expected
    ):
        cube_.dimension_types = dim_types
        cube_.ndim = len(dim_types)

        s = _BaseCubeResultMatrix._get_regular_matrix_counts_slice(cube_, slice_idx)

        assert s == expected

    @pytest.mark.parametrize(
        ("counts", "counts_slice", "expected"),
        (
            ([[1, 2, 3], [4, 5, 6]], np.s_[:], [[1, 2, 3], [4, 5, 6]]),
            ([[1, 2, 3], [4, 5, 6]], np.s_[0], [1, 2, 3]),
            ([[1, 2, 3], [4, 5, 6]], np.s_[0, 0], 1),
        ),
    )
    def it_knows_how_to_get_sliced_counts_to_help(
        self, request, cube_, counts, counts_slice, expected
    ):
        counts = np.array(counts)
        cube_.counts = counts
        cube_.unweighted_counts = counts
        cube_.counts_with_missings = counts
        fake_slice_idx = 1
        _get_regular_matrix_counts_slice = method_mock(
            request,
            _BaseCubeResultMatrix,
            "_get_regular_matrix_counts_slice",
            return_value=counts_slice,
        )

        sliced_counts = _BaseCubeResultMatrix._get_sliced_counts(cube_, fake_slice_idx)

        counts, unweighted, with_missing = sliced_counts
        assert counts.tolist() == expected
        assert unweighted.tolist() == expected
        _get_regular_matrix_counts_slice.assert_called_once_with(cube_, fake_slice_idx)

    @pytest.mark.parametrize(
        ("dimension_types", "expected"),
        (
            ((DT.MR, DT.MR), _MrXMrMatrix),
            ((DT.MR, DT.CAT), _MrXCatMatrix),
            ((DT.CAT, DT.MR), _CatXMrMatrix),
            ((DT.CAT, DT.CAT), _CatXCatMatrix),
        ),
    )
    def it_knows_its_regular_matrix_factory_class(self, dimension_types, expected):
        matrix_class = _BaseCubeResultMatrix._get_regular_matrix_factory_class(
            dimension_types
        )

        assert matrix_class == expected

    @pytest.mark.parametrize(
        ("matrix_class_name",),
        (
            ("_CatXCatMatrix",),
            ("_MrXCatMatrix",),
            ("_CatXMrMatrix",),
            ("_MrXMrMatrix",),
        ),
    )
    def it_creates_matrix_using_factory_to_help(
        self, request, cube_, dimension_, matrix_class_name
    ):
        cube_.dimension_types = (DT.CAT, DT.MR, DT.CAT)
        MatrixCls_ = class_mock(request, "cr.cube.matrix.%s" % matrix_class_name)
        _get_regular_matrix_factory_class = method_mock(
            request,
            _BaseCubeResultMatrix,
            "_get_regular_matrix_factory_class",
            return_value=MatrixCls_,
        )
        _get_sliced_counts = method_mock(
            request,
            _BaseCubeResultMatrix,
            "_get_sliced_counts",
            return_value=([[1], [2]], [[3], [4]]),
        )

        matrix = _BaseCubeResultMatrix._regular_matrix_factory(
            cube_, (dimension_, dimension_), slice_idx=17
        )

        _get_regular_matrix_factory_class.assert_called_once_with((DT.MR, DT.CAT))
        _get_sliced_counts.assert_called_once_with(cube_, 17)
        MatrixCls_.assert_called_once_with(
            (dimension_, dimension_), [[1], [2]], [[3], [4]]
        )
        assert matrix is MatrixCls_.return_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)
