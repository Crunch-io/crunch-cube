# encoding: utf-8

"""Unit test suite for `cr.cube.matrix.cubemeasure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension, _ValidElements
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.matrix.cubemeasure import (
    _ArrXArrCubeCounts,
    _ArrXCatCubeCounts,
    _ArrXMrCubeCounts,
    _BaseCubeCounts,
    _BaseCubeMeans,
    _BaseCubeMeasure,
    _BaseCubeOverlaps,
    _BaseCubeStdDev,
    _BaseCubeSums,
    _BaseUnconditionalCubeCounts,
    _CatXArrCubeCounts,
    _CatXCatCubeCounts,
    _CatXCatCubeMeans,
    _CatXCatCubeStdDev,
    _CatXCatCubeSums,
    _CatXCatUnconditionalCubeCounts,
    _CatXMrCubeCounts,
    _CatXMrCubeMeans,
    _CatXMrCubeStdDev,
    _CatXMrCubeSums,
    _CatXMrUnconditionalCubeCounts,
    CubeMeasures,
    _MrXArrCubeCounts,
    _MrXCatCubeCounts,
    _MrXCatCubeMeans,
    _MrXCatCubeStdDev,
    _MrXCatCubeSums,
    _MrXCatUnconditionalCubeCounts,
    _MrXMrCubeCounts,
    _MrXMrCubeMeans,
    _MrXMrCubeStdDev,
    _MrXMrCubeSums,
    _MrXMrUnconditionalCubeCounts,
)

from ...unitutil import class_mock, instance_mock, method_mock, property_mock


class DescribeCubeMeasures:
    """Unit test suite for `cr.cube.matrix.cubemeasure.CubeMeasures` object."""

    @pytest.mark.parametrize(
        "cube_measure_, CubeMeasureCls",
        (
            ("cube_means", _BaseCubeMeans),
            ("cube_overlaps", _BaseCubeOverlaps),
            ("cube_sum", _BaseCubeSums),
            ("cube_stddev", _BaseCubeStdDev),
        ),
    )
    def it_provides_access_to_cube_measures_object(
        self, request, cube_, dimensions_, cube_measure_, CubeMeasureCls
    ):
        _cube_measure_ = instance_mock(request, CubeMeasureCls)
        CubeMeasureCls_ = class_mock(
            request,
            "cr.cube.matrix.cubemeasure.%s" % CubeMeasureCls.__name__,
        )
        CubeMeasureCls_.factory.return_value = _cube_measure_
        cube_measures = CubeMeasures(cube_, dimensions_, slice_idx=37)

        cube_measure = getattr(cube_measures, cube_measure_)

        CubeMeasureCls_.factory.assert_called_once_with(cube_, dimensions_, 37)
        assert cube_measure is _cube_measure_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimensions_(self, request):
        return (instance_mock(request, Dimension), instance_mock(request, Dimension))


class Describe_BaseCubeMeasure:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseCubeMeasure`."""

    @pytest.mark.parametrize(
        "ndim, tab_dim_type, expected_value",
        (
            (2, None, slice(None, None, None)),
            (3, DT.MR, (3, 0)),
            (3, DT.CAT, 3),
        ),
    )
    def it_computes_index_expression_that_selects_measure_values_from_cube_to_help(
        self, request, ndim, tab_dim_type, expected_value
    ):
        cube_ = instance_mock(request, Cube, ndim=ndim, dimension_types=(tab_dim_type,))
        measure = _BaseCubeMeasure(None)

        assert measure._slice_idx_expr(cube_, slice_idx=3) == expected_value


# === COUNTS (UNWEIGHTED & WEIGHTED)


class Describe_BaseCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseCubeCounts`."""

    @pytest.mark.parametrize(
        ("dimension_types", "CubeCountsCls"),
        (
            ((DT.MR, DT.MR), _MrXMrCubeCounts),
            ((DT.MR, DT.CAT), _MrXCatCubeCounts),
            ((DT.CAT, DT.MR), _CatXMrCubeCounts),
            ((DT.CAT, DT.CAT), _CatXCatCubeCounts),
            ((DT.NUM_ARRAY, DT.MR), _ArrXMrCubeCounts),
            ((DT.NUM_ARRAY, DT.CAT), _ArrXCatCubeCounts),
            ((DT.CA_CAT, DT.CA_SUBVAR), _CatXArrCubeCounts),
            ((DT.MR, DT.CA_SUBVAR), _MrXArrCubeCounts),
        ),
    )
    def it_provides_a_factory_for_constructing_cube_count_objects(
        self,
        request,
        cube_,
        dimensions_,
        dimension_types,
        CubeCountsCls,
    ):
        cube_counts_ = instance_mock(request, CubeCountsCls)
        CubeCountsCls_ = class_mock(
            request,
            "cr.cube.matrix.cubemeasure.%s" % CubeCountsCls.__name__,
            return_value=cube_counts_,
        )
        _slice_idx_expr_ = method_mock(
            request,
            _BaseCubeCounts,
            "_slice_idx_expr",
            return_value=1,
            autospec=False,
        )
        cube_.dimension_types = dimension_types

        cube_counts = _BaseCubeCounts.factory(
            [[1, 2], [3, 4]], "diff_nans", cube_, dimensions_, slice_idx=7
        )

        _slice_idx_expr_.assert_called_once_with(cube_, 7)
        CubeCountsCls_.assert_called_once_with(dimensions_, [3, 4], "diff_nans")
        assert cube_counts is cube_counts_

    def it_provides_columns_pruning_mask(self, request):
        property_mock(
            request,
            _BaseCubeCounts,
            "_columns_pruning_base",
            return_value=np.array([0, 2, 0, 4]),
        )
        cube_counts = _BaseCubeCounts(None, None, None)

        assert cube_counts.columns_pruning_mask.tolist() == [True, False, True, False]

    def it_provides_rows_pruning_mask(self, request):
        property_mock(
            request,
            _BaseCubeCounts,
            "_rows_pruning_base",
            return_value=np.array([0, 2, 0, 4]),
        )
        cube_counts = _BaseCubeCounts(None, None, None)

        assert cube_counts.rows_pruning_mask.tolist() == [True, False, True, False]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimensions_(self, request):
        return (
            instance_mock(request, Dimension),
            instance_mock(request, Dimension),
        )


class Describe_ArrXArrCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._ArrXArrCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(raw_counts)

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base is None

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base is None

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(raw_counts)

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(raw_counts)

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base is None

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base is None

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(raw_counts)

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array([4.5, 3.8, 4.9])
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _ArrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(np.array([6.7, 6.5]))

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 3) np.float64 ndarray of counts as received from Cube."""
        return np.array(
            [
                [1.1, 2.3, 3.3],
                [3.4, 1.5, 1.6],
            ]
        )


class Describe_ArrXCatCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._ArrXCatCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(raw_counts)

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base is None

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base is None

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(raw_counts)

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            np.array(
                [
                    [6.7, 6.7, 6.7],
                    [6.5, 6.5, 6.5],
                ]
            )
        )

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base == pytest.approx([6.7, 6.5])

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base == pytest.approx([6.7, 6.5])

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            np.array(
                [
                    [6.7, 6.7, 6.7],
                    [6.5, 6.5, 6.5],
                ]
            )
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array([4.5, 3.8, 4.9])
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _ArrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(np.array([20.1, 19.5]))

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 3) np.float64 ndarray of counts as received from Cube."""
        return np.array(
            [
                [1.1, 2.3, 3.3],
                [3.4, 1.5, 1.6],
            ]
        )


class Describe_ArrXMrCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._ArrXMrCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            np.array(
                [
                    [1.1, 2.2, 3.3],
                    [4.4, 5.5, 6.6],
                ]
            )
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base is None

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base is None

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(
            np.array(
                [
                    [1.1, 2.2, 3.3],
                    [4.4, 5.5, 6.6],
                ]
            )
        )

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            np.array(
                [
                    [4.8, 6.8, 8.8],
                    [5.9, 7.9, 9.9],
                ]
            )
        )

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base is None

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base is None

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            np.array(
                [
                    [4.8, 6.8, 8.8],
                    [5.9, 7.9, 9.9],
                ]
            )
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array(
                # --- sel/not for all rows in a column.
                # --- example: 10.7 == 1.1 + 3.7 + 4.4 + 1.5
                [10.7, 14.7, 18.7]
            )
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _ArrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(np.array([20.4, 23.7]))

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 3, 2) np.float ndarray of weighted cube-counts as received from Cube."""
        return np.array(
            [
                # -- sel / not ---------------
                [  # -- row 0 ----------------
                    [1.1, 3.7],  # -- col 0 --
                    [2.2, 4.6],  # -- col 1 --
                    [3.3, 5.5],  # -- col 2 --
                ],
                [  # -- row 1 ----------------
                    [4.4, 1.5],  # -- col 0 --
                    [5.5, 2.4],  # -- col 1 --
                    [6.6, 3.3],  # -- col 2 --
                ],
            ]
        )


class Describe_CatXArrCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXArrCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            np.array(
                [
                    [4.5, 3.8, 4.9],
                    [4.5, 3.8, 4.9],
                ],
            )
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base == pytest.approx(np.array([4.5, 3.8, 4.9]))

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base == pytest.approx(
            np.array([4.5, 3.8, 4.9])
        )

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(raw_counts)

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(raw_counts)

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base is None

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base is None

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            np.array(
                [
                    [4.5, 3.8, 4.9],
                    [4.5, 3.8, 4.9],
                ],
            )
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array([9.0, 7.6, 9.8])
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _CatXArrCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(np.array([6.7, 6.5]))

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 3) np.float64 ndarray of counts as received from Cube."""
        return np.array(
            [
                [1.1, 2.3, 3.3],
                [3.4, 1.5, 1.6],
            ]
        )


class Describe_CatXCatCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXCatCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            np.array(
                [
                    [4.5, 3.8, 4.9],
                    [4.5, 3.8, 4.9],
                ]
            )
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base == pytest.approx(np.array([4.5, 3.8, 4.9]))

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base == pytest.approx(
            np.array([13.2, 13.2, 13.2])
        )

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(raw_counts)

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            np.array(
                [
                    [6.7, 6.7, 6.7],
                    [6.5, 6.5, 6.5],
                ]
            )
        )

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base == pytest.approx(np.array([6.7, 6.5]))

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base == pytest.approx(np.array([13.2, 13.2]))

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base == pytest.approx(np.array([13.2, 13.2]))

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            np.array(
                [
                    [13.2, 13.2, 13.2],
                    [13.2, 13.2, 13.2],
                ]
            )
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array([9.0, 7.6, 9.8])
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _CatXCatCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(np.array([20.1, 19.5]))

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 3) np.float64 ndarray of counts as received from Cube."""
        return np.array(
            [
                [1.1, 2.3, 3.3],
                [3.4, 1.5, 1.6],
            ]
        )


class Describe_CatXMrCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXMrCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            np.array([[5.5, 7.7, 9.9], [5.5, 7.7, 9.9]])
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base == pytest.approx(np.array([5.5, 7.7, 9.9]))

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base == pytest.approx(
            np.array([10.7, 14.7, 18.7])
        )

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(
            np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        )

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            np.array([[4.8, 6.8, 8.8], [5.9, 7.9, 9.9]])
        )

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base is None

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base is None

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            np.array([[10.7, 14.7, 18.7], [10.7, 14.7, 18.7]])
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array(
                # --- sel/not for all rows in a column.
                # --- example: 10.7 == 1.1 + 3.7 + 4.4 + 1.5
                [10.7, 14.7, 18.7]
            )
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _CatXMrCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(np.array([20.4, 23.7]))

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 2, 3) np.float ndarray of weighted cube-counts as received from Cube."""
        return np.array(
            [
                # -- sel / not ---------------
                [  # -- row 0 ----------------
                    [1.1, 3.7],  # -- col 0 --
                    [2.2, 4.6],  # -- col 1 --
                    [3.3, 5.5],  # -- col 2 --
                ],
                [  # -- row 1 ----------------
                    [4.4, 1.5],  # -- col 0 --
                    [5.5, 2.4],  # -- col 1 --
                    [6.6, 3.3],  # -- col 2 --
                ],
            ]
        )


class Describe_MrXArrCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXArrCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            np.array([[4.5, 6.5, 8.5], [5.9, 7.9, 9.9]])
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base is None

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base is None

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(
            np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        )

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        )

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base is None

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base is None

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            np.array([[4.5, 6.5, 8.5], [5.9, 7.9, 9.9]])
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array([10.4, 14.4, 18.4])
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _MrXArrCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(
            # --- sel/not for all columns in a row
            # --- example: 19.5 = 1.1 + 2.2 + 3.3 + 3.4 + 4.3 + 5.2
            np.array([19.5, 23.7])
        )

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 2, 3) np.float ndarray of weighted cube-counts as received from Cube."""
        return np.array(
            [  # --   0    1    2  cols ------
                [  # -- row 0 ----------------
                    [1.1, 2.2, 3.3],  # -- sel
                    [3.4, 4.3, 5.2],  # -- not
                ],
                [  # -- row 1 ----------------
                    [4.4, 5.5, 6.6],  # -- sel
                    [1.5, 2.4, 3.3],  # -- not
                ],
            ]
        )


class Describe_MrXCatCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXCatCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            np.array([[4.5, 6.5, 8.5], [5.9, 7.9, 9.9]])
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base is None

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base is None

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(
            np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        )

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            np.array([[6.6, 6.6, 6.6], [16.5, 16.5, 16.5]])
        )

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base == pytest.approx(np.array([6.6, 16.5]))

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base == pytest.approx(np.array([19.5, 23.7]))

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            np.array([[19.5, 19.5, 19.5], [23.7, 23.7, 23.7]])
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            np.array([10.4, 14.4, 18.4])
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _MrXCatCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(
            # --- sel/not for all columns in a row
            # --- example: 19.5 = 1.1 + 2.2 + 3.3 + 3.4 + 4.3 + 5.2
            np.array([19.5, 23.7])
        )

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 3) np.float64 ndarray of counts as received from Cube."""
        return np.array(
            [  # --   0    1    2  cols ------
                [  # -- row 0 ----------------
                    [1.1, 2.2, 3.3],  # -- sel
                    [3.4, 4.3, 5.2],  # -- not
                ],
                [  # -- row 1 ----------------
                    [4.4, 5.5, 6.6],  # -- sel
                    [1.5, 2.4, 3.3],  # -- not
                ],
            ]
        )


class Describe_MrXMrCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXMrCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            # --- sel/not for each column in sel row
            # --- example: 4.6 = 1.1 + 3.5
            np.array([[4.6, 8.0], [4.2, 12.6]])
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base is None

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base is None

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(np.array([[1.1, 3.1], [2.8, 7.3]]))

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            # --- sel/not for each column in sel row
            # --- example: 3.4 = 1.1 + 2.3
            np.array([[3.4, 7.1], [6.3, 15.5]])
        )

    def it_knows_its_rows_base(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_base is None

    def it_knows_its_rows_table_base(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.rows_table_base is None

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_base is None

    def it_knows_its_table_bases(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.table_bases == pytest.approx(
            # --- sel/not for all columns in sel/not rows
            # --- example: 12.2 = 1.1+ 2.3 + 3.5+ 5.3
            np.array([[12.2, 18.8], [14.1, 30.6]])
        )

    def it_provides_the_columns_pruning_base_to_help(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts._columns_pruning_base == pytest.approx(
            # --- sel/not for all rows in a selected column
            # --- example: 8.8 = 1.1 + 3.5 + 2.8 + 1.4
            np.array([8.8, 20.6])
        )

    def it_provides_the_rows_pruning_base_to_help(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts._rows_pruning_base == pytest.approx(
            # --- sel/not for all columns in a selected row
            # --- example: 10.5 = 1.1 + 2.3 + 3.1 + 4.0
            np.array([10.5, 21.8])
        )

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 2, 2, 2) np.float ndarray of cube-counts as from Cube."""
        return np.array(
            [  # ------ sel/not (col) -----
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [1.1, 2.3],  # -- col 0
                        [3.1, 4.0],  # -- col 1
                    ],
                    [  # -- not selected --
                        [3.5, 5.3],  # -- col 0
                        [4.9, 6.8],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [2.8, 3.5],  # -- col 0
                        [7.3, 8.2],  # -- col 1
                    ],
                    [  # -- not selected --
                        [1.4, 6.4],  # -- col 0
                        [5.3, 9.8],  # -- col 1
                    ],
                ],
            ]
        )


# === MEANS ===


class Describe_BaseCubeMeans:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseCubeMeans`."""

    @pytest.mark.parametrize(
        "dimension_types, CubeMeansCls",
        (
            ((DT.MR, DT.MR), _MrXMrCubeMeans),
            ((DT.MR, DT.CAT), _MrXCatCubeMeans),
            ((DT.CAT, DT.MR), _CatXMrCubeMeans),
            ((DT.CAT, DT.CAT), _CatXCatCubeMeans),
        ),
    )
    def it_provides_a_factory_for_constructing_cube_means_objects(
        self, request, dimension_types, CubeMeansCls
    ):
        cube_ = instance_mock(request, Cube)
        dimensions_ = (
            instance_mock(request, Dimension),
            instance_mock(request, Dimension),
        )
        cube_means_ = instance_mock(request, CubeMeansCls)
        CubeMeanCls_ = class_mock(
            request,
            "cr.cube.matrix.cubemeasure.%s" % CubeMeansCls.__name__,
            return_value=cube_means_,
        )
        _slice_idx_expr_ = method_mock(
            request,
            _BaseCubeMeans,
            "_slice_idx_expr",
            return_value=1,
            autospec=False,
        )
        cube_.dimension_types = dimension_types
        cube_.means = [[1, 2], [3, 4]]

        cube_means = _BaseCubeMeans.factory(cube_, dimensions_, slice_idx=7)

        _slice_idx_expr_.assert_called_once_with(cube_, 7)
        CubeMeanCls_.assert_called_once_with(dimensions_, [3, 4])
        assert cube_means is cube_means_

    def but_it_raises_a_value_error_when_cube_result_does_not_contain_mean_measure(
        self, cube_
    ):
        cube_.means = None

        with pytest.raises(ValueError) as e:
            _BaseCubeMeans.factory(cube_, None, None)

        assert str(e.value) == "cube-result does not contain cube-means measure"

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)


class Describe_CatXCatCubeMeans:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXCatCubeMeans`."""

    def it_knows_its_means(self, raw_means):
        cube_means = _CatXCatCubeMeans(None, raw_means)

        assert cube_means.means.tolist() == [
            [1.1, 2.3, 3.3],
            [3.4, 1.5, 1.6],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_means(self):
        """(2, 3) np.float64 ndarray of means as received from Cube."""
        return np.array(
            [
                [1.1, 2.3, 3.3],
                [3.4, 1.5, 1.6],
            ]
        )


class Describe_CatXMrCubeMeans:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXMrCubeMeans`."""

    def it_knows_its_means(self, raw_means):
        cube_means = _CatXMrCubeMeans(None, raw_means)

        assert cube_means.means.tolist() == [
            [1.1, 2.2, 3.2],
            [4.3, 5.1, 6.1],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_means(self):
        """(2, 3, 2) np.float64 ndarray of means as received from Cube."""
        return np.array(
            [  # -- axes are (rows, cols, sel/not) --
                # --sel/not--
                [  # -- row 0 ------------
                    [1.1, 6.1],  # -- col 0 --
                    [2.2, 5.2],  # -- col 1 --
                    [3.2, 4.2],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4.3, 3.1],  # -- col 0 --
                    [5.1, 2.1],  # -- col 1 --
                    [6.1, 1.1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )


class Describe_MrXCatCubeMeans:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXCatCubeMeans`."""

    def it_knows_its_means(self, raw_means):
        cube_means = _MrXCatCubeMeans(None, raw_means)

        assert cube_means.means.tolist() == [
            [1.1, 6.1],
            [4.3, 3.1],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_means(self):
        """(2, 3, 2) np.int float64 of means as received from Cube."""
        return np.array(
            [  # -- axes are (rows, cols, sel/not) --
                # --sel/not--
                [  # -- row 0 ------------
                    [1.1, 6.1],  # -- col 0 --
                    [2.2, 5.2],  # -- col 1 --
                    [3.2, 4.2],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4.3, 3.1],  # -- col 0 --
                    [5.1, 2.1],  # -- col 1 --
                    [6.1, 1.1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )


class Describe_MrXMrCubeMeans:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXMrCubeMeans`."""

    def it_knows_its_means(self, raw_means):
        cube_means = _MrXMrCubeMeans(None, raw_means)

        assert cube_means.means.tolist() == [
            [0.1, 0.1],
            [0.4, 0.5],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_means(self):
        """(2, 2, 2, 2) np.float64 ndarray of means as from Cube."""
        return np.array(
            # -- axes are (rows, sel/not, cols, sel/not) --
            [
                [  # -- row 0 -------------
                    # --sel/not--
                    [  # -- selected ------
                        [0.1, 0.8],  # -- col 0
                        [0.1, 0.7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [0.2, 0.6],  # -- col 0
                        [0.3, 0.5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [0.4, 0.4],  # -- col 0
                        [0.5, 0.3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [0.6, 0.2],  # -- col 0
                        [0.7, 0.1],  # -- col 1
                    ],
                ],
            ]
        )


# === STD DEV ===


class Describe_BaseCubeStdDev:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseCubeStdDev`."""

    @pytest.mark.parametrize(
        "dimension_types, CubeStdDevCls",
        (
            ((DT.MR, DT.MR), _MrXMrCubeStdDev),
            ((DT.MR, DT.CAT), _MrXCatCubeStdDev),
            ((DT.CAT, DT.MR), _CatXMrCubeStdDev),
            ((DT.CAT, DT.CAT), _CatXCatCubeStdDev),
        ),
    )
    def it_provides_a_factory_for_constructing_cube_stddev_objects(
        self, request, dimension_types, CubeStdDevCls, cube_
    ):
        dimensions_ = (
            instance_mock(request, Dimension),
            instance_mock(request, Dimension),
        )
        cube_stddev_ = instance_mock(request, CubeStdDevCls)
        CubeStdDevCls_ = class_mock(
            request,
            "cr.cube.matrix.cubemeasure.%s" % CubeStdDevCls.__name__,
            return_value=cube_stddev_,
        )
        _slice_idx_expr_ = method_mock(
            request,
            _BaseCubeStdDev,
            "_slice_idx_expr",
            return_value=1,
            autospec=False,
        )
        cube_.dimension_types = dimension_types
        cube_.stddev = [[1, 2], [3, 4]]

        cube_stddev = _BaseCubeStdDev.factory(cube_, dimensions_, slice_idx=7)

        _slice_idx_expr_.assert_called_once_with(cube_, 7)
        CubeStdDevCls_.assert_called_once_with(dimensions_, [3, 4])
        assert cube_stddev is cube_stddev_

    def but_it_raises_a_value_error_when_cube_result_does_not_contain_stddev_measure(
        self, cube_
    ):
        cube_.stddev = None

        with pytest.raises(ValueError) as e:
            _BaseCubeStdDev.factory(cube_, None, None)

        assert str(e.value) == "cube-result does not contain cube-stddev measure"

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)


class Describe_CatXCatCubeStdDev:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXCatCubeStdDev`."""

    def it_knows_its_stddev(self, raw_stddev):
        cube_stddev = _CatXCatCubeStdDev(None, raw_stddev)

        assert cube_stddev.stddev.tolist() == [
            [1.1, 2.3, 3.3],
            [3.4, 1.5, 1.6],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_stddev(self):
        """(2, 3) np.float64 ndarray of stddev as received from Cube."""
        return np.array(
            [
                [1.1, 2.3, 3.3],
                [3.4, 1.5, 1.6],
            ]
        )


class Describe_CatXMrCubeStdDev:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXMrCubeStdDev`."""

    def it_knows_its_stddev(self, raw_stddev):
        cube_stddev = _CatXMrCubeStdDev(None, raw_stddev)

        assert cube_stddev.stddev.tolist() == [
            [1.1, 2.2, 3.2],
            [4.3, 5.1, 6.1],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_stddev(self):
        """(2, 3, 2) np.float64 ndarray of stddev as received from Cube."""
        return np.array(
            [  # -- axes are (rows, cols, sel/not) --
                # --sel/not--
                [  # -- row 0 ------------
                    [1.1, 6.1],  # -- col 0 --
                    [2.2, 5.2],  # -- col 1 --
                    [3.2, 4.2],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4.3, 3.1],  # -- col 0 --
                    [5.1, 2.1],  # -- col 1 --
                    [6.1, 1.1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )


class Describe_MrXCatCubeStdDev:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXCatCubeStdDev`."""

    def it_knows_its_stddev(self, raw_stddev):
        cube_stddev = _MrXCatCubeStdDev(None, raw_stddev)

        assert cube_stddev.stddev.tolist() == [
            [1.1, 6.1],
            [4.3, 3.1],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_stddev(self):
        """(2, 3, 2) np.int float64 of stddev as received from Cube."""
        return np.array(
            [  # -- axes are (rows, cols, sel/not) --
                # --sel/not--
                [  # -- row 0 ------------
                    [1.1, 6.1],  # -- col 0 --
                    [2.2, 5.2],  # -- col 1 --
                    [3.2, 4.2],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4.3, 3.1],  # -- col 0 --
                    [5.1, 2.1],  # -- col 1 --
                    [6.1, 1.1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )


class Describe_MrXMrCubeStdDev:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXMrCubeStdDev`."""

    def it_knows_its_stddev(self, raw_stddev):
        cube_stddev = _MrXMrCubeStdDev(None, raw_stddev)

        assert cube_stddev.stddev.tolist() == [
            [0.1, 0.1],
            [0.4, 0.5],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_stddev(self):
        """(2, 2, 2, 2) np.float64 ndarray of stddev as from Cube."""
        return np.array(
            # -- axes are (rows, sel/not, cols, sel/not) --
            [
                [  # -- row 0 -------------
                    # --sel/not--
                    [  # -- selected ------
                        [0.1, 0.8],  # -- col 0
                        [0.1, 0.7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [0.2, 0.6],  # -- col 0
                        [0.3, 0.5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [0.4, 0.4],  # -- col 0
                        [0.5, 0.3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [0.6, 0.2],  # -- col 0
                        [0.7, 0.1],  # -- col 1
                    ],
                ],
            ]
        )


# === SUM ===


class Describe_BaseCubeSum:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseCubeSum`."""

    @pytest.mark.parametrize(
        "dimension_types, CubeSumCls",
        (
            ((DT.MR, DT.MR), _MrXMrCubeSums),
            ((DT.MR, DT.CAT), _MrXCatCubeSums),
            ((DT.CAT, DT.MR), _CatXMrCubeSums),
            ((DT.CAT, DT.CAT), _CatXCatCubeSums),
        ),
    )
    def it_provides_a_factory_for_constructing_cube_sum_objects(
        self, request, dimension_types, CubeSumCls, cube_
    ):
        dimensions_ = (
            instance_mock(request, Dimension),
            instance_mock(request, Dimension),
        )
        cube_sums_ = instance_mock(request, CubeSumCls)
        CubeSumCls_ = class_mock(
            request,
            "cr.cube.matrix.cubemeasure.%s" % CubeSumCls.__name__,
            return_value=cube_sums_,
        )
        _slice_idx_expr_ = method_mock(
            request,
            _BaseCubeSums,
            "_slice_idx_expr",
            return_value=1,
            autospec=False,
        )
        cube_.dimension_types = dimension_types
        cube_.sums = [[1, 2], [3, 4]]

        cube_sums = _BaseCubeSums.factory(cube_, dimensions_, slice_idx=7)

        _slice_idx_expr_.assert_called_once_with(cube_, 7)
        CubeSumCls_.assert_called_once_with(dimensions_, [3, 4])
        assert cube_sums is cube_sums_

    def but_it_raises_a_value_error_when_cube_result_does_not_contain_sum_measure(
        self, cube_
    ):
        cube_.sums = None

        with pytest.raises(ValueError) as e:
            _BaseCubeSums.factory(cube_, None, None)

        assert str(e.value) == "cube-result does not contain cube-sum measure"

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)


class Describe_CatXCatCubeSums:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXCatCubeSums`."""

    def it_knows_its_sum(self, raw_sums):
        cube_sum = _CatXCatCubeSums(None, raw_sums)

        assert cube_sum.sums.tolist() == [
            [1.1, 2.3, 3.3],
            [3.4, 1.5, 1.6],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_sums(self):
        """(2, 3) np.float64 ndarray of sums as received from Cube."""
        return np.array(
            [
                [1.1, 2.3, 3.3],
                [3.4, 1.5, 1.6],
            ]
        )


class Describe_CatXMrCubeSum:
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXMrCubeSums`."""

    def it_knows_its_sum(self, raw_sums):
        cube_sum = _CatXMrCubeSums(None, raw_sums)

        assert cube_sum.sums.tolist() == [
            [1.1, 2.2, 3.2],
            [4.3, 5.1, 6.1],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_sums(self):
        """(2, 3, 2) np.float64 ndarray of sums as received from Cube."""
        return np.array(
            [  # -- axes are (rows, cols, sel/not) --
                # --sel/not--
                [  # -- row 0 ------------
                    [1.1, 6.1],  # -- col 0 --
                    [2.2, 5.2],  # -- col 1 --
                    [3.2, 4.2],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4.3, 3.1],  # -- col 0 --
                    [5.1, 2.1],  # -- col 1 --
                    [6.1, 1.1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )


class Describe_MrXCatCubeSum:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXCatCubeSums`."""

    def it_knows_its_sum(self, raw_sums):
        cube_sum = _MrXCatCubeSums(None, raw_sums)

        assert cube_sum.sums.tolist() == [
            [1.1, 6.1],
            [4.3, 3.1],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_sums(self):
        """(2, 3, 2) np.int float64 of sums as received from Cube."""
        return np.array(
            [  # -- axes are (rows, cols, sel/not) --
                # --sel/not--
                [  # -- row 0 ------------
                    [1.1, 6.1],  # -- col 0 --
                    [2.2, 5.2],  # -- col 1 --
                    [3.2, 4.2],  # -- col 2 --
                ],
                [  # -- row 1 ------------
                    [4.3, 3.1],  # -- col 0 --
                    [5.1, 2.1],  # -- col 1 --
                    [6.1, 1.1],  # -- col 2 --
                    # --------------------
                ],
            ]
        )


class Describe_MrXMrCubeSum:
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXMrCubeSums`."""

    def it_knows_its_sums(self, raw_sums):
        cube_sum = _MrXMrCubeSums(None, raw_sums)

        assert cube_sum.sums.tolist() == [
            [0.1, 0.1],
            [0.4, 0.5],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_sums(self):
        """(2, 2, 2, 2) np.float64 ndarray of sums as from Cube."""
        return np.array(
            # -- axes are (rows, sel/not, cols, sel/not) --
            [
                [  # -- row 0 -------------
                    # --sel/not--
                    [  # -- selected ------
                        [0.1, 0.8],  # -- col 0
                        [0.1, 0.7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [0.2, 0.6],  # -- col 0
                        [0.3, 0.5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [0.4, 0.4],  # -- col 0
                        [0.5, 0.3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [0.6, 0.2],  # -- col 0
                        [0.7, 0.1],  # -- col 1
                    ],
                ],
            ]
        )


# === (WEIGHTED) UNCONDITIONAL COUNTS ===


class Describe_BaseUnconditionalCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseUnconditionalCubeCounts`."""

    @pytest.mark.parametrize(
        (
            "dimension_types",
            "UnconditinoalCubeCountsCls",
            "counts_with_missings",
            "expected_counts_with_missings",
        ),
        (
            (
                (DT.MR, DT.MR),
                _MrXMrUnconditionalCubeCounts,
                [[1, 2], [3, 4]],
                [3, 4],
            ),
            (
                (DT.MR, DT.CAT),
                _MrXCatUnconditionalCubeCounts,
                [[1, 2], [3, 4]],
                [3, 4],
            ),
            (
                (DT.CAT, DT.MR),
                _CatXMrUnconditionalCubeCounts,
                [[1, 2], [3, 4]],
                [3, 4],
            ),
            (
                (DT.CAT, DT.CAT),
                _CatXCatUnconditionalCubeCounts,
                [[1, 2], [3, 4]],
                [3, 4],
            ),
        ),
    )
    def it_provides_a_factory_for_constructing_weighted_cube_count_objects(
        self,
        request,
        cube_,
        dimensions_,
        dimension_types,
        UnconditinoalCubeCountsCls,
        counts_with_missings,
        expected_counts_with_missings,
    ):
        uncond_cube_counts_ = instance_mock(request, UnconditinoalCubeCountsCls)
        WeightedCubeCountsCls_ = class_mock(
            request,
            "cr.cube.matrix.cubemeasure.%s" % UnconditinoalCubeCountsCls.__name__,
            return_value=uncond_cube_counts_,
        )
        _slice_idx_expr_ = method_mock(
            request,
            _BaseUnconditionalCubeCounts,
            "_slice_idx_expr",
            return_value=1,
            autospec=False,
        )
        cube_.dimension_types = dimension_types
        cube_.counts_with_missings = counts_with_missings

        weighted_cube_counts = _BaseUnconditionalCubeCounts.factory(
            cube_,
            dimensions_,
            slice_idx=2,
        )

        _slice_idx_expr_.assert_called_with(cube_, 2)
        WeightedCubeCountsCls_.assert_called_once_with(
            dimensions_,
            expected_counts_with_missings,
        )
        assert weighted_cube_counts is uncond_cube_counts_

    def it_provides_valid_row_idxs_to_help(self, request, dimensions_):
        valid_elements_ = instance_mock(request, _ValidElements, element_idxs=(0, 2, 4))
        dimensions_[0].valid_elements = valid_elements_
        unconditional_counts = _BaseUnconditionalCubeCounts(dimensions_, None)

        assert unconditional_counts._valid_row_idxs[0].tolist() == [0, 2, 4]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)
