# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.cubemeasure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _ValidElements
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.matrix.cubemeasure import (
    BaseCubeResultMatrix,
    _CatXCatMatrix,
    _CatXCatMeansMatrix,
    _CatXMrMatrix,
    _CatXMrMeansMatrix,
    _MrXCatMatrix,
    _MrXCatMeansMatrix,
    _MrXMrMatrix,
)

from ...unitutil import class_mock, instance_mock, method_mock, property_mock


class DescribeBaseCubeResultMatrix(object):
    """Unit test suite for `cr.cube.matrix.BaseCubeResultMatrix` object."""

    @pytest.mark.parametrize(
        "has_means, factory_method_name",
        ((True, "_means_matrix_factory"), (False, "_regular_matrix_factory")),
    )
    def it_calls_the_correct_factory_method_for_appropriate_matrix_type(
        self, request, cube_, dimension_, has_means, factory_method_name
    ):
        cube_.has_means = has_means
        cube_result_matrix_ = instance_mock(request, BaseCubeResultMatrix)
        factory_method = method_mock(
            request,
            BaseCubeResultMatrix,
            factory_method_name,
            return_value=cube_result_matrix_,
        )

        cube_result_matrix = BaseCubeResultMatrix.factory(
            cube_, (dimension_, dimension_), slice_idx=71
        )

        factory_method.assert_called_once_with(cube_, (dimension_, dimension_), 71)
        assert cube_result_matrix is cube_result_matrix_

    def it_knows_its_column_proportions(self, request):
        property_mock(
            request,
            BaseCubeResultMatrix,
            "weighted_counts",
            return_value=np.array([[1, 2, 3], [4, 5, 6]]),
        )
        property_mock(
            request,
            BaseCubeResultMatrix,
            "columns_margin",
            return_value=np.array([5, 7, 9]),
        )
        matrix = BaseCubeResultMatrix(None, None, None)

        np.testing.assert_almost_equal(
            matrix.column_proportions,
            np.array([[0.2, 0.2857143, 0.3333333], [0.8, 0.7142857, 0.6666667]]),
        )

    def it_knows_its_columns_dimension(self, dimension_):
        matrix = BaseCubeResultMatrix([None, dimension_], None, None)
        assert matrix.columns_dimension == dimension_

    def it_knows_its_rows_dimension(self, dimension_):
        matrix = BaseCubeResultMatrix([dimension_, None], None, None)
        assert matrix.rows_dimension == dimension_

    def it_can_compute_array_type_std_residual_to_help(self):
        counts = np.array([[0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22]])
        total = np.array([51, 63, 75, 87])
        rowsum = np.array([[1, 5, 9, 13], [17, 21, 25, 29], [33, 37, 41, 45]])
        colsum = np.array([24, 30, 36, 42])
        matrix = BaseCubeResultMatrix(None, None, None)

        residuals = matrix._array_type_std_res(counts, total, rowsum, colsum)

        np.testing.assert_almost_equal(
            residuals,
            np.array(
                [
                    [-0.9521905, -0.3555207, -0.2275962, -0.1660169],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.2762585, 0.1951985, 0.1485686, 0.1184429],
                ]
            ),
        )

    def but_it_produces_zero_valued_zscores_for_a_deficient_matrix(self):
        counts = np.array([[0, 2], [0, 10]])
        matrix = BaseCubeResultMatrix(None, None, None)

        residuals = matrix._array_type_std_res(counts, None, None, None)

        assert residuals.tolist() == [[0, 0], [0, 0]]

    @pytest.mark.parametrize(
        "dimension_types, matrix_class_name",
        (
            ((DT.MR, DT.CAT), "_MrXCatMeansMatrix"),
            ((DT.CAT, DT.MR), "_CatXMrMeansMatrix"),
            ((DT.CAT, DT.CAT), "_CatXCatMeansMatrix"),
        ),
    )
    def it_can_construct_a_means_matrix_for_a_2D_slice_to_help(
        self, request, cube_, dimension_types, dimension_, matrix_class_name
    ):
        cube_.dimension_types = dimension_types
        cube_.ndim = 2
        cube_.counts = [1, 2, 3, 4]
        cube_.unweighted_counts = [5, 6, 7, 8]
        MatrixCls_ = class_mock(
            request, "cr.cube.matrix.cubemeasure.%s" % matrix_class_name
        )

        matrix = BaseCubeResultMatrix._means_matrix_factory(
            cube_, (dimension_, dimension_), None
        )

        MatrixCls_.assert_called_once_with(
            (dimension_, dimension_), [1, 2, 3, 4], [5, 6, 7, 8]
        )
        assert matrix is MatrixCls_.return_value

    @pytest.mark.parametrize(
        "dimension_types, matrix_class_name",
        (
            ((None, DT.MR, DT.CAT), "_MrXCatMeansMatrix"),
            ((None, DT.CAT, DT.MR), "_CatXMrMeansMatrix"),
            ((None, DT.CAT, DT.CAT), "_CatXCatMeansMatrix"),
        ),
    )
    def and_it_can_construct_a_means_matrix_for_a_3D_slice_to_help(
        self, request, cube_, dimension_types, dimension_, matrix_class_name
    ):
        cube_.dimension_types = dimension_types
        cube_.ndim = 3
        cube_.counts = [None, [1, 2, 3, 4], None]
        cube_.unweighted_counts = [None, [5, 6, 7, 8], None]
        MatrixCls_ = class_mock(
            request, "cr.cube.matrix.cubemeasure.%s" % matrix_class_name
        )

        matrix = BaseCubeResultMatrix._means_matrix_factory(
            cube_, (dimension_, dimension_), slice_idx=1
        )

        MatrixCls_.assert_called_once_with(
            (dimension_, dimension_), [1, 2, 3, 4], [5, 6, 7, 8]
        )
        assert matrix is MatrixCls_.return_value

    def but_it_raises_on_MEANS_MR_X_MR(self, cube_):
        cube_.dimension_types = (DT.MR, DT.MR)

        with pytest.raises(NotImplementedError) as e:
            BaseCubeResultMatrix._means_matrix_factory(cube_, None, None)

        assert str(e.value) == "MR x MR with means is not implemented"

    @pytest.mark.parametrize(
        "dimension_types, expected_value",
        (
            ((DT.MR, DT.MR), _MrXMrMatrix),
            ((DT.MR, DT.CAT), _MrXCatMatrix),
            ((DT.CAT, DT.MR), _CatXMrMatrix),
            ((DT.CAT, DT.CAT), _CatXCatMatrix),
        ),
    )
    def it_knows_its_regular_matrix_class_to_help(
        self, dimension_types, expected_value
    ):
        assert (
            BaseCubeResultMatrix._regular_matrix_class(dimension_types)
            is expected_value
        )

    @pytest.mark.parametrize(
        ("slice_idx", "dim_types", "expected"),
        (
            # --- <= 2D ---
            (0, (DT.CAT,), np.s_[:]),
            (1, (DT.CAT,), np.s_[:]),
            (0, (DT.CAT, DT.CAT), np.s_[:]),
            (1, (DT.CAT, DT.CAT), np.s_[:]),
            # --- 3D, no MR as tabs ---
            (0, (DT.CAT, DT.CAT, DT.CAT), np.s_[0]),
            (1, (DT.CAT, DT.CAT, DT.CAT), np.s_[1]),
            (2, (DT.CAT, DT.CAT, DT.CAT), np.s_[2]),
            # --- 3D, MR as tabs ---
            (0, (DT.MR, DT.CAT, DT.CAT), np.s_[0, 0]),
            (1, (DT.MR, DT.CAT, DT.CAT), np.s_[1, 0]),
            (2, (DT.MR, DT.CAT, DT.CAT), np.s_[2, 0]),
        ),
    )
    def it_knows_its_regular_matrix_counts_slice_to_help(
        self, cube_, slice_idx, dim_types, expected
    ):
        cube_.dimension_types = dim_types
        cube_.ndim = len(dim_types)

        s = BaseCubeResultMatrix._regular_matrix_counts_slice(cube_, slice_idx)

        assert s == expected

    @pytest.mark.parametrize(
        "matrix_class_name",
        (
            "_CatXCatMatrix",
            "_MrXCatMatrix",
            "_CatXMrMatrix",
            "_MrXMrMatrix",
        ),
    )
    def it_can_construct_a_regular_matrix_to_help(
        self, request, cube_, dimension_, matrix_class_name
    ):
        cube_.dimension_types = (DT.CAT, DT.MR, DT.CAT)
        MatrixCls_ = class_mock(
            request, "cr.cube.matrix.cubemeasure.%s" % matrix_class_name
        )
        _regular_matrix_class = method_mock(
            request,
            BaseCubeResultMatrix,
            "_regular_matrix_class",
            return_value=MatrixCls_,
        )
        _sliced_counts = method_mock(
            request,
            BaseCubeResultMatrix,
            "_sliced_counts",
            return_value=([[1], [2]], [[3], [4]]),
        )

        matrix = BaseCubeResultMatrix._regular_matrix_factory(
            cube_, (dimension_, dimension_), slice_idx=17
        )

        _regular_matrix_class.assert_called_once_with((DT.MR, DT.CAT))
        _sliced_counts.assert_called_once_with(cube_, 17)
        MatrixCls_.assert_called_once_with(
            (dimension_, dimension_), [[1], [2]], [[3], [4]]
        )
        assert matrix is MatrixCls_.return_value

    @pytest.mark.parametrize(
        ("counts", "counts_slice", "expected"),
        (
            ([[1, 2, 3], [4, 5, 6]], np.s_[:], [[1, 2, 3], [4, 5, 6]]),
            ([[1, 2, 3], [4, 5, 6]], np.s_[0], [1, 2, 3]),
            ([[1, 2, 3], [4, 5, 6]], np.s_[0, 0], 1),
        ),
    )
    def it_knows_its_sliced_counts_to_help(
        self, request, cube_, counts, counts_slice, expected
    ):
        counts = np.array(counts)
        cube_.counts = counts
        cube_.unweighted_counts = counts
        cube_.counts_with_missings = counts
        _regular_matrix_counts_slice = method_mock(
            request,
            BaseCubeResultMatrix,
            "_regular_matrix_counts_slice",
            return_value=counts_slice,
        )

        sliced_counts = BaseCubeResultMatrix._sliced_counts(cube_, slice_idx=23)

        _regular_matrix_counts_slice.assert_called_once_with(cube_, 23)
        counts, unweighted, with_missing = sliced_counts
        assert counts.tolist() == expected
        assert unweighted.tolist() == expected

    def it_produces_a_valid_row_indexer_to_help(self, request, dimension_):
        dimension_.valid_elements = instance_mock(
            request, _ValidElements, element_idxs=(0, 1, 2)
        )
        matrix = BaseCubeResultMatrix((dimension_, None), None, None)

        valid_row_idxs = matrix._valid_row_idxs

        assert isinstance(valid_row_idxs, tuple)
        assert len(valid_row_idxs) == 1
        assert valid_row_idxs[0].tolist() == [0, 1, 2]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)


class Describe_CatXCatMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXCatMatrix` object."""

    def it_knows_its_columns_index(self, request):
        property_mock(
            request,
            _CatXCatMatrix,
            "column_proportions",
            return_value=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        )
        property_mock(
            request,
            _CatXCatMatrix,
            "_baseline",
            return_value=np.array([[0.2], [0.8]]),
        )
        matrix = _CatXCatMatrix(None, None, None)

        np.testing.assert_almost_equal(
            matrix.column_index, np.array([[50.0, 100.0, 150.0], [50.0, 62.5, 75.0]])
        )

    def it_knows_its_columns_base(self):
        matrix = _CatXCatMatrix(None, None, np.array([[1, 2, 3], [4, 5, 6]]))
        assert matrix.columns_base.tolist() == [5, 7, 9]

    def it_knows_its_columns_margin(self):
        matrix = _CatXCatMatrix(None, np.array([[1, 2, 3], [4, 5, 6]]), None)
        assert matrix.columns_margin.tolist() == [5, 7, 9]

    @pytest.mark.parametrize(
        ("unweighted_counts", "expected"),
        (
            # --- 1 "row", 0 cols ---
            ([[]], []),
            # --- 1 row, 1 col ---
            ([[1]], [1]),
            # --- 1 row, 3 cols ---
            ([[1, 2, 3]], [1, 2, 3]),
            # --- 3 rows, 0 cols ---
            ([[], [], []], []),
            # --- 3 rows, 1 col ---
            ([[1], [2], [3]], [6]),
            # --- 3 rows, 3 cols ---
            ([[1, 2, 3], [4, 5, 6]], [5, 7, 9]),
        ),
    )
    def it_knows_its_columns_pruning_base(self, unweighted_counts, expected):
        matrix = _CatXCatMatrix(None, None, np.array(unweighted_counts))

        columns_pruning_base = matrix.columns_pruning_base

        assert columns_pruning_base.shape == (len(expected),)
        assert columns_pruning_base.tolist() == expected

    @pytest.mark.parametrize(
        ("unweighted_counts", "expected"),
        (
            ([[1, 2, 3]], [6]),
            ([[1, 2, 3], [4, 5, 6]], [6, 15]),
            ([[1], [2], [3]], [1, 2, 3]),
        ),
    )
    def it_knows_its_rows_base(self, unweighted_counts, expected):
        matrix = _CatXCatMatrix(None, None, unweighted_counts)
        assert matrix.rows_base.tolist() == expected

    def it_knows_its_rows_margin(self):
        weighted_counts = np.array([[1, 2, 3], [4, 5, 6]])
        matrix = _CatXCatMatrix(None, weighted_counts, None)

        assert matrix.rows_margin.tolist() == [6, 15]

    @pytest.mark.parametrize(
        ("unweighted_counts", "expected"),
        (
            ([[1, 2, 3]], [6]),
            ([[1, 2, 3], [4, 5, 6]], [6, 15]),
            ([[1], [2], [3]], [1, 2, 3]),
        ),
    )
    def it_knows_its_rows_pruning_base(self, unweighted_counts, expected):
        matrix = _CatXCatMatrix(None, None, unweighted_counts)
        assert matrix.rows_pruning_base.tolist() == expected

    def it_knows_its_table_base(self):
        unweighted_counts = np.array([[1, 2, 3], [4, 5, 6]])
        assert _CatXCatMatrix(None, None, unweighted_counts).table_base == 21

    def it_knows_its_table_margin(self):
        weighted_counts = np.array([[1, 2, 3], [4, 5, 6]])
        assert _CatXCatMatrix(None, weighted_counts, None).table_margin == 21

    def it_knows_its_table_stderrs(self, request):
        property_mock(
            request,
            _CatXCatMatrix,
            "_table_proportion_variances",
            return_value=np.array([[0.5, 1.0], [0.25, 2.0]]),
        )
        property_mock(request, _CatXCatMatrix, "table_margin", return_value=2.0)

        np.testing.assert_almost_equal(
            _CatXCatMatrix(None, None, None).table_stderrs,
            np.array([[0.5, 0.7071068], [0.3535534, 1.0]]),
        )

    def it_knows_its_unweighted_counts(self):
        unweighted_cube_counts = np.array([[1, 2, 3], [4, 5, 6]])
        matrix = _CatXCatMatrix(None, None, unweighted_cube_counts)

        assert matrix.unweighted_counts.tolist() == [[1, 2, 3], [4, 5, 6]]

    def it_knows_its_weighted_counts(self):
        weighted_cube_counts = np.array([[3, 2, 1], [6, 5, 4]])
        matrix = _CatXCatMatrix(None, weighted_cube_counts, None)

        assert matrix.weighted_counts.tolist() == [[3, 2, 1], [6, 5, 4]]

    def it_knows_its_zscores(self):
        weighted_counts = np.array([[3, 2, 1], [6, 5, 4]])
        matrix = _CatXCatMatrix(None, weighted_counts, None)

        np.testing.assert_almost_equal(
            matrix.zscores,
            np.array([[0.41833001, 0.0, -0.48605555], [-0.41833001, 0.0, 0.48605555]]),
        )

    @pytest.mark.parametrize(
        "weighted_counts, expected_value",
        (
            (
                np.array([[1, 2], [0, 0]]),
                np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            ),
            (
                np.array([[0, 2], [0, 4]]),
                np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            ),
        ),
    )
    def but_its_zscores_are_NaNs_for_a_deficient_matrix(
        self, weighted_counts, expected_value
    ):
        np.testing.assert_almost_equal(
            _CatXCatMatrix(None, weighted_counts, None).zscores, expected_value
        )

    def it_knows_its_baseline_to_help(self, request):
        property_mock(
            request, _CatXCatMatrix, "_valid_row_idxs", return_value=np.array([0, 1])
        )
        counts_with_missings = np.array([[1, 2, 3], [4, 5, 6]])

        np.testing.assert_almost_equal(
            _CatXCatMatrix(None, None, None, counts_with_missings)._baseline,
            np.array([[0.2857143], [0.7142857]]),
        )

    def it_knows_its_table_proportion_variances_to_help(self, request):
        weighted_counts = np.arange(6).reshape(2, 3)
        np.testing.assert_almost_equal(
            _CatXCatMatrix(None, weighted_counts, None)._table_proportion_variances,
            np.array([[0.0, 0.0622222, 0.1155556], [0.16, 0.1955556, 0.2222222]]),
        )


class Describe_CatXCatMeansMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXCatMeansMatrix` object."""

    def it_knows_its_means(self):
        cube_means = np.array([[2, 3, 1], [5, 6, 4]])
        matrix = _CatXCatMeansMatrix(None, cube_means, None)

        assert matrix.means.tolist() == [[2, 3, 1], [5, 6, 4]]

    def it_knows_its_weighted_counts(self):
        cube_means = np.array([[3, 2, 1], [6, 5, 4]])
        matrix = _CatXCatMeansMatrix(None, cube_means, None)

        np.testing.assert_equal(
            matrix.weighted_counts,
            [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        )


class Describe_CatXMrMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXMrMatrix` object."""

    def it_knows_its_columns_pruning_base(self):
        unweighted_cube_counts = np.array(
            [
                [[1, 6], [2, 5], [3, 5]],  # --- row 0 ---
                [[5, 3], [6, 3], [7, 2]],  # --- row 1 ---
            ]
        )
        matrix = _CatXMrMatrix(None, None, unweighted_cube_counts, None)

        assert matrix.columns_pruning_base.tolist() == [15, 16, 17]

    def it_knows_its_rows_base(self):
        unweighted_cube_counts = np.array(
            [[[1, 6], [2, 5], [3, 4]], [[5, 3], [6, 2], [7, 1]]]
        )
        matrix = _CatXMrMatrix(None, None, unweighted_cube_counts, None)

        assert matrix.rows_base.tolist() == [[7, 7, 7], [8, 8, 8]]

    def it_knows_its_rows_margin(self):
        weighted_cube_counts = np.array(
            [[[1, 6], [2, 5], [3, 4]], [[5, 3], [6, 2], [7, 1]]]
        )
        matrix = _CatXMrMatrix(None, weighted_cube_counts, None, None)

        assert matrix.rows_margin.tolist() == [[7, 7, 7], [8, 8, 8]]

    def it_knows_its_rows_pruning_base(self):
        unweighted_cube_counts = np.array(
            [[[1, 6], [2, 5], [3, 4]], [[5, 3], [6, 2], [7, 1]]]
        )
        matrix = _CatXMrMatrix(None, None, unweighted_cube_counts, None)

        assert matrix.rows_pruning_base.tolist() == [21, 24]

    def it_knows_its_table_base(self):
        unweighted_cube_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 6],  # -- col 0 --
                    [2, 5],  # -- col 1 --
                    [3, 4],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4, 3],  # -- col 0 --
                    [5, 2],  # -- col 1 --
                    [6, 1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )
        matrix = _CatXMrMatrix(None, None, unweighted_cube_counts, None)

        assert matrix.table_base.tolist() == [14, 14, 14]

    def it_knows_its_table_margin(self):
        weighted_cube_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 6],  # -- col 0 --
                    [2, 5],  # -- col 1 --
                    [3, 4],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4, 3],  # -- col 0 --
                    [5, 2],  # -- col 1 --
                    [6, 1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )
        matrix = _CatXMrMatrix(None, weighted_cube_counts, None, None)

        assert matrix.table_margin.tolist() == [14, 14, 14]

    def it_knows_its_unweighted_counts(self):
        unweighted_cube_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 6],  # -- col 0 --
                    [2, 5],  # -- col 1 --
                    [3, 4],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4, 3],  # -- col 0 --
                    [5, 2],  # -- col 1 --
                    [6, 1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )
        matrix = _CatXMrMatrix(None, None, unweighted_cube_counts, None)

        assert matrix.unweighted_counts.tolist() == [[1, 2, 3], [4, 5, 6]]

    def it_knows_its_weighted_counts(self):
        weighted_cube_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 6],  # -- col 0 --
                    [2, 5],  # -- col 1 --
                    [3, 4],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4, 3],  # -- col 0 --
                    [5, 2],  # -- col 1 --
                    [6, 1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )
        matrix = _CatXMrMatrix(None, weighted_cube_counts, None, None)

        assert matrix.weighted_counts.tolist() == [[1, 2, 3], [4, 5, 6]]

    def it_knows_its_zscores(self, request):
        _array_type_std_res_ = method_mock(
            request,
            _CatXMrMatrix,
            "_array_type_std_res",
            return_value=np.array([[1, 2], [3, 4]]),
        )
        weighted_cube_counts = np.arange(24).reshape(3, 4, 2)
        matrix = _CatXMrMatrix(None, weighted_cube_counts, None)

        zscores = matrix.zscores

        self_, counts, total, rowsum, colsum = _array_type_std_res_.call_args.args
        assert self_ is matrix
        assert counts.tolist() == [[0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22]]
        assert total.tolist() == [51, 63, 75, 87]
        assert rowsum.tolist() == [[1, 5, 9, 13], [17, 21, 25, 29], [33, 37, 41, 45]]
        assert colsum.tolist() == [24, 30, 36, 42]
        assert zscores.tolist() == [[1, 2], [3, 4]]

    def it_knows_its_baseline_to_help(self, request):
        property_mock(
            request, _CatXMrMatrix, "_valid_row_idxs", return_value=np.array([0, 1])
        )
        counts_with_missings = np.array(
            [[[1, 6], [2, 5], [3, 4]], [[4, 3], [5, 2], [6, 1]]]
        )

        np.testing.assert_almost_equal(
            _CatXMrMatrix(None, None, None, counts_with_missings)._baseline,
            np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        )

    def it_knows_its_table_proportion_variances_to_help(self, request):
        weighted_cube_counts = np.arange(12).reshape(2, 3, 2)
        np.testing.assert_almost_equal(
            _CatXMrMatrix(None, weighted_cube_counts, None)._table_proportion_variances,
            np.array([[0.0, 0.0826446, 0.1155556], [0.244898, 0.231405, 0.2222222]]),
        )


class Describe_CatXMrMeansMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXMrMeansMatrix` object."""

    def it_knows_its_means(self):
        means = np.array([[[1, 6], [2, 5], [3, 4]], [[5, 3], [6, 2], [7, 1]]])
        np.testing.assert_equal(
            _CatXMrMeansMatrix(None, means, None).means,
            np.array([[1, 2, 3], [5, 6, 7]]),
        )


class Describe_MrXCatMatrix(object):
    """Unit test suite for `cr.cube.matrix._MrXCatMatrix` object."""

    def it_knows_its_columns_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [3, 2, 1],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts, None).columns_base,
            np.array([[5, 7, 9], [10, 10, 10]]),
        )

    def it_knows_its_columns_margin(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [3, 2, 1],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, weighted_counts, None, None).columns_margin,
            np.array([[5, 7, 9], [10, 10, 10]]),
        )

    def it_knows_its_columns_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [3, 2, 1],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts, None).columns_pruning_base,
            np.array([15, 17, 19]),
        )

    def it_knows_its_rows_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 2, 3],  # -- selected --
                    [4, 5, 6],  # -- not --
                ],
                [  # -- row 1 ------------
                    [7, 8, 9],  # -- selected --
                    [3, 2, 1],  # -- not --
                    # --------------------
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts, None).rows_pruning_base,
            np.array([21, 30]),
        )

    def it_knows_its_table_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [0, 4, 2],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts).table_base, [21, 30]
        )

    def it_knows_its_table_margin(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [0, 4, 2],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, weighted_counts, None).table_margin, [21, 30]
        )

    def it_knows_its_table_stderrs(self, request):
        property_mock(
            request,
            _MrXCatMatrix,
            "_table_proportion_variances",
            return_value=np.array([[0.5, 1.0], [0.25, 2.0]]),
        )
        property_mock(
            request,
            _MrXCatMatrix,
            "table_margin",
            return_value=np.array([[1, 2], [3, 4]]),
        )

        np.testing.assert_almost_equal(
            _MrXCatMatrix(None, None, None).table_stderrs,
            np.array(
                [
                    [[0.7071068, 0.7071068], [0.5, 1.0]],
                    [[0.4082483, 0.5], [0.2886751, 0.7071068]],
                ]
            ),
        )

    def it_knows_its_unweighted_counts(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ------------
                    [1, 2, 3],  # -- selected --
                    [4, 5, 6],  # -- not --
                ],
                [  # -- row 1 ------------
                    [7, 8, 9],  # -- selected --
                    [0, 4, 2],  # -- not --
                    # --------------------
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, None, unweighted_counts).unweighted_counts,
            np.array([[1, 2, 3], [7, 8, 9]]),
        )

    def it_knows_its_weighted_counts(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [1, 2, 3],  # -- selected
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ---------------
                    [7, 8, 9],  # -- selected
                    [0, 4, 2],  # -- not
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXCatMatrix(None, weighted_counts, None).weighted_counts,
            np.array([[1, 2, 3], [7, 8, 9]]),
        )

    def it_knows_its_zscores(self, request):
        _array_type_std_res_ = method_mock(
            request, _MrXCatMatrix, "_array_type_std_res", return_value=[[1, 2], [3, 4]]
        )
        weighted_counts = np.arange(24).reshape(3, 2, 4)
        matrix = _MrXCatMatrix(None, weighted_counts, None)

        zscores = matrix.zscores

        self_, counts, total, rowsum, colsum = _array_type_std_res_.call_args.args
        assert self_ is matrix
        np.testing.assert_equal(
            counts, np.array([[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]])
        )
        np.testing.assert_equal(total, np.array([[28], [92], [156]]))
        np.testing.assert_equal(rowsum, np.array([[6], [38], [70]]))
        np.testing.assert_equal(
            colsum, np.array([[4, 6, 8, 10], [20, 22, 24, 26], [36, 38, 40, 42]])
        )
        assert zscores == [[1, 2], [3, 4]]

    def it_knows_its_baseline_to_help(self, request):
        property_mock(
            request, _MrXCatMatrix, "_valid_row_idxs", return_value=np.array([0, 1])
        )
        counts_with_missings = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 4, 2]]]
        )

        np.testing.assert_almost_equal(
            _MrXCatMatrix(None, None, None, counts_with_missings)._baseline,
            np.array([[0.2857143], [0.8]]),
        )

    def it_knows_its_table_proportion_variances_to_help(self, request):
        weighted_counts = np.arange(12).reshape(2, 2, 3)
        np.testing.assert_almost_equal(
            _MrXCatMatrix(None, weighted_counts, None)._table_proportion_variances,
            np.array([[0.0, 0.0622222, 0.1155556], [0.1038062, 0.118416, 0.1322568]]),
        )


class Describe_MrXCatMeansMatrix(object):
    """Unit test suite for `cr.cube.matrix._MrXCatMeansMatrix` object."""

    def it_knows_its_means(self):
        means = np.arange(24).reshape(3, 2, 4)
        np.testing.assert_equal(
            _MrXCatMeansMatrix(None, means, None).means,
            np.array([[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]]),
        )


class Describe_MrXMrMatrix(object):
    """Unit test suite for `cr.cube.matrix._MrXMrMatrix` object."""

    def it_knows_its_columns_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).columns_base,
            np.array([[2, 8, 4], [10, 4, 13]]),
        )

    def it_knows_its_columns_margin(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, weighted_counts, None, None).columns_margin,
            np.array([[2, 8, 4], [10, 4, 13]]),
        )

    def it_knows_its_columns_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).columns_pruning_base,
            np.array([12, 12, 17]),
        )

    def it_knows_its_rows_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).rows_base,
            np.array([[8, 9, 8], [8, 8, 11]]),
        )

    def it_knows_its_rows_margin(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, weighted_counts, None, None).rows_margin,
            np.array([[8, 9, 8], [8, 8, 11]]),
        )

    def it_knows_its_rows_pruning_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 ---------------
                    [[0, 8], [2, 7], [1, 7]],
                    [[2, 6], [6, 8], [3, 5]],
                ],
                [  # -- row 1 ---------------
                    [[4, 4], [1, 7], [8, 3]],
                    [[6, 2], [3, 5], [5, 2]],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).rows_pruning_base,
            np.array([25, 27]),
        )

    def it_knows_its_table_base(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [0, 5],  # -- col 0
                        [1, 4],  # -- col 1
                    ],
                    [  # -- not selected --
                        [2, 3],  # -- col 0
                        [3, 1],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [4, 1],  # -- col 0
                        [5, 0],  # -- col 1
                    ],
                    [  # -- not selected --
                        [6, 9],  # -- col 0
                        [7, 6],  # -- col 1
                    ],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).table_base,
            np.array([[10, 9], [20, 18]]),
        )

    def it_knows_its_table_margin(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [0, 5],  # -- col 0
                        [1, 4],  # -- col 1
                    ],
                    [  # -- not selected --
                        [2, 3],  # -- col 0
                        [3, 1],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [4, 1],  # -- col 0
                        [5, 0],  # -- col 1
                    ],
                    [  # -- not selected --
                        [6, 9],  # -- col 0
                        [7, 6],  # -- col 1
                    ],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, weighted_counts, None, None).table_margin,
            np.array([[10, 9], [20, 18]]),
        )

    def it_knows_its_unweighted_counts(self):
        unweighted_counts = np.array(
            [
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [0, 8],  # -- col 0
                        [1, 7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [2, 6],  # -- col 0
                        [3, 5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [4, 4],  # -- col 0
                        [5, 3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [6, 2],  # -- col 0
                        [7, 1],  # -- col 1
                    ],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, None, unweighted_counts, None).unweighted_counts,
            np.array([[0, 1], [4, 5]]),
        )

    def it_knows_its_weighted_counts(self):
        weighted_counts = np.array(
            [
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [0, 8],  # -- col 0
                        [1, 7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [2, 6],  # -- col 0
                        [3, 5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [4, 4],  # -- col 0
                        [5, 3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [6, 2],  # -- col 0
                        [7, 1],  # -- col 1
                    ],
                ],
            ]
        )
        np.testing.assert_equal(
            _MrXMrMatrix(None, weighted_counts, None, None).weighted_counts,
            np.array([[0, 1], [4, 5]]),
        )

    def it_knows_its_zscores(self, request):
        _array_type_std_res_ = method_mock(
            request, _MrXMrMatrix, "_array_type_std_res", return_value=[[1, 2], [3, 4]]
        )
        weighted_counts = np.arange(48).reshape(3, 2, 4, 2)
        matrix = _MrXMrMatrix(None, weighted_counts, None)

        zscores = matrix.zscores

        self_, counts, total, rowsum, colsum = _array_type_std_res_.call_args.args
        assert self_ is matrix
        np.testing.assert_equal(
            counts, np.array([[0, 2, 4, 6], [16, 18, 20, 22], [32, 34, 36, 38]])
        )
        np.testing.assert_equal(
            total, np.array([[18, 26, 34, 42], [82, 90, 98, 106], [146, 154, 162, 170]])
        )
        np.testing.assert_equal(
            rowsum, np.array([[1, 5, 9, 13], [33, 37, 41, 45], [65, 69, 73, 77]])
        )
        np.testing.assert_equal(
            colsum, np.array([[8, 12, 16, 20], [40, 44, 48, 52], [72, 76, 80, 84]])
        )
        assert zscores == [[1, 2], [3, 4]]

    def it_knows_its_baseline_to_help(self, request):
        property_mock(
            request, _MrXMrMatrix, "_valid_row_idxs", return_value=np.array([0, 1])
        )
        counts_with_missings = np.array(
            [
                [[[0, 8], [1, 7]], [[2, 6], [3, 5]]],
                [[[4, 4], [5, 3]], [[6, 2], [7, 1]]],
            ]
        )

        np.testing.assert_almost_equal(
            _MrXMrMatrix(None, None, None, counts_with_missings)._baseline,
            np.array([[0.5, 0.5], [0.5, 0.5]]),
        )

    def it_knows_its_table_proportion_variances_to_help(self, request):
        weighted_counts = np.arange(24).reshape(2, 2, 3, 2)
        np.testing.assert_almost_equal(
            _MrXMrMatrix(None, weighted_counts, None)._table_proportion_variances,
            np.array([[0.0, 0.0826446, 0.1155556], [0.1560874, 0.16, 0.1630506]]),
        )
