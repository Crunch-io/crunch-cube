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
    BaseCubeResultMatrix,
    _BaseUnconditionalCubeCounts,
    _BaseUnweightedCubeCounts,
    _CatXArrCubeCounts,
    _CatXCatCubeCounts,
    _CatXCatCubeMeans,
    _CatXCatCubeStdDev,
    _CatXCatCubeSums,
    _CatXCatMatrix,
    _CatXCatUnconditionalCubeCounts,
    _CatXCatUnweightedCubeCounts,
    _CatXMrCubeCounts,
    _CatXMrCubeMeans,
    _CatXMrCubeStdDev,
    _CatXMrCubeSums,
    _CatXMrMatrix,
    _CatXMrUnconditionalCubeCounts,
    _CatXMrUnweightedCubeCounts,
    CubeMeasures,
    _MrXArrCubeCounts,
    _MrXCatCubeCounts,
    _MrXCatCubeMeans,
    _MrXCatCubeStdDev,
    _MrXCatCubeSums,
    _MrXCatMatrix,
    _MrXCatUnconditionalCubeCounts,
    _MrXCatUnweightedCubeCounts,
    _MrXMrCubeCounts,
    _MrXMrCubeMeans,
    _MrXMrCubeStdDev,
    _MrXMrCubeSums,
    _MrXMrMatrix,
    _MrXMrUnconditionalCubeCounts,
    _MrXMrUnweightedCubeCounts,
    _NumArrayXMrUnweightedCubeCounts,
    _NumArrayXCatUnweightedCubeCounts,
)

from ...unitutil import class_mock, instance_mock, method_mock, property_mock


class DescribeCubeMeasures(object):
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


class Describe_BaseCubeMeasure(object):
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


class Describe_BaseCubeCounts(object):
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


class Describe_ArrXArrCubeCounts(object):
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


class Describe_ArrXCatCubeCounts(object):
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


class Describe_ArrXMrCubeCounts(object):
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


class Describe_CatXArrCubeCounts(object):
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


class Describe_CatXCatCubeCounts(object):
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


class Describe_CatXMrCubeCounts(object):
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


class Describe_MrXArrCubeCounts(object):
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


class Describe_MrXCatCubeCounts(object):
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


class Describe_MrXMrCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXMrCubeCounts`."""

    def it_knows_its_column_bases(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.column_bases == pytest.approx(
            np.array([[2.2, 4.4], [11.0, 13.2]])
        )

    def it_knows_its_columns_base(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_base is None

    def it_knows_its_columns_table_base(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.columns_table_base is None

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.counts == pytest.approx(np.array([[0.0, 1.1], [4.4, 5.5]]))

    def it_knows_its_row_bases(self, raw_counts):
        cube_counts = _MrXMrCubeCounts(None, raw_counts, None)

        assert cube_counts.row_bases == pytest.approx(
            np.array([[8.8, 8.8], [8.8, 8.8]])
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
            np.array([[17.6, 17.6], [17.6, 17.6]])
        )

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(2, 2, 2, 2) np.float ndarray of cube-counts as from Cube."""
        return np.array(
            [  # ------ sel/not (col) -----
                [  # -- row 0 -------------
                    [  # -- selected ------
                        [0.0, 8.8],  # -- col 0
                        [1.1, 7.7],  # -- col 1
                    ],
                    [  # -- not selected --
                        [2.2, 6.6],  # -- col 0
                        [3.3, 5.5],  # -- col 1
                    ],
                ],
                [  # -- row 1 -------------
                    [  # -- selected ------
                        [4.4, 4.4],  # -- col 0
                        [5.5, 3.3],  # -- col 1
                    ],
                    [  # -- not selected --
                        [6.6, 2.2],  # -- col 0
                        [7.7, 1.1],  # -- col 1
                    ],
                ],
            ]
        )


# === MEANS ===


class Describe_BaseCubeMeans(object):
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


class Describe_CatXCatCubeMeans(object):
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


class Describe_CatXMrCubeMeans(object):
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


class Describe_MrXCatCubeMeans(object):
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


class Describe_MrXMrCubeMeans(object):
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


class Describe_BaseCubeStdDev(object):
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


class Describe_CatXCatCubeStdDev(object):
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


class Describe_CatXMrCubeStdDev(object):
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


class Describe_MrXCatCubeStdDev(object):
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


class Describe_MrXMrCubeStdDev(object):
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


class Describe_BaseCubeSum(object):
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


class Describe_CatXCatCubeSums(object):
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


class Describe_CatXMrCubeSum(object):
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


class Describe_MrXCatCubeSum(object):
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


class Describe_MrXMrCubeSum(object):
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


class Describe_BaseUnconditionalCubeCounts(object):
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

        unconditional_counts._valid_row_idxs == np.array([0, 2, 4])

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimensions_(self, request):
        return instance_mock(request, Dimension), instance_mock(request, Dimension)


# === UNWEIGHTED COUNTS ===


class Describe_BaseUnweightedCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseUnweightedCubeCounts`."""

    @pytest.mark.parametrize(
        (
            "dimension_types",
            "UnweightedCubeCountsCls",
            "unweighted_counts",
            "unweighted_valid_counts",
            "diff_nans",
            "expected_counts",
        ),
        (
            (
                (DT.MR, DT.MR),
                _MrXMrUnweightedCubeCounts,
                [[1, 2], [3, 4]],
                None,
                False,
                [3, 4],
            ),
            (
                (DT.MR, DT.CAT),
                _MrXCatUnweightedCubeCounts,
                [[1, 2], [3, 4]],
                None,
                False,
                [3, 4],
            ),
            (
                (DT.CAT, DT.MR),
                _CatXMrUnweightedCubeCounts,
                [[1, 2], [3, 4]],
                None,
                False,
                [3, 4],
            ),
            (
                (DT.CAT, DT.CAT),
                _CatXCatUnweightedCubeCounts,
                [[1, 2], [3, 4]],
                None,
                False,
                [3, 4],
            ),
            (
                (DT.NUM_ARRAY, DT.MR),
                _NumArrayXMrUnweightedCubeCounts,
                [[1, 2], [3, 4]],
                [[1, 2], [6, 7]],
                True,
                [6, 7],
            ),
            (
                (DT.NUM_ARRAY, DT.CAT),
                _NumArrayXCatUnweightedCubeCounts,
                [[1, 2], [3, 4]],
                [[1, 2], [6, 7]],
                True,
                [6, 7],
            ),
        ),
    )
    def it_provides_a_factory_for_constructing_unweighted_cube_count_objects(
        self,
        request,
        dimension_types,
        UnweightedCubeCountsCls,
        unweighted_counts,
        unweighted_valid_counts,
        diff_nans,
        expected_counts,
    ):
        cube_ = instance_mock(request, Cube)
        dimensions_ = (
            instance_mock(request, Dimension),
            instance_mock(request, Dimension),
        )
        unweighted_cube_counts_ = instance_mock(request, UnweightedCubeCountsCls)
        UnweightedCubeCountsCls_ = class_mock(
            request,
            "cr.cube.matrix.cubemeasure.%s" % UnweightedCubeCountsCls.__name__,
            return_value=unweighted_cube_counts_,
        )
        _slice_idx_expr_ = method_mock(
            request,
            _BaseUnweightedCubeCounts,
            "_slice_idx_expr",
            return_value=1,
            autospec=False,
        )
        cube_.dimension_types = dimension_types
        cube_.unweighted_valid_counts = unweighted_valid_counts
        cube_.unweighted_counts = unweighted_counts

        unweighted_cube_counts = _BaseUnweightedCubeCounts.factory(
            cube_, dimensions_, slice_idx=7
        )

        _slice_idx_expr_.assert_called_once_with(cube_, 7)
        UnweightedCubeCountsCls_.assert_called_once_with(
            dimensions_, expected_counts, diff_nans
        )
        assert unweighted_cube_counts is unweighted_cube_counts_

    @pytest.mark.parametrize(
        "columns_base, expected_value",
        (
            (np.arange(3), [[0, 1, 2], [0, 1, 2]]),
            (np.arange(6).reshape(2, 3), [[0, 1, 2], [3, 4, 5]]),
        ),
    )
    def it_computes_the_column_bases(self, request, columns_base, expected_value):
        property_mock(
            request,
            _BaseUnweightedCubeCounts,
            "columns_base",
            return_value=columns_base,
        )
        property_mock(
            request,
            _BaseUnweightedCubeCounts,
            "unweighted_counts",
            return_value=np.array([[0, 0, 0], [0, 0, 0]]),
        )
        unweighted_cube_counts = _BaseUnweightedCubeCounts(None, None, None)

        assert unweighted_cube_counts.column_bases.tolist() == expected_value


class Describe_CatXCatUnweightedCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXCatUnweightedCubeCounts`."""

    def it_knows_its_columns_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_base.tolist() == [5, 7, 9]

    def it_knows_its_columns_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_pruning_base.tolist() == [5, 7, 9]

    def it_knows_its_row_bases(self, request, raw_unweighted_counts):
        property_mock(
            request,
            _CatXCatUnweightedCubeCounts,
            "rows_base",
            return_value=np.array([2, 1]),
        )
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )

        assert unweighted_cube_counts.row_bases.tolist() == [[2, 2, 2], [1, 1, 1]]

    def it_knows_its_rows_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_pruning_base.tolist() == [6, 15]

    def it_knows_its_table_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.table_base == 21

    def it_knows_its_table_bases(self, request, raw_unweighted_counts):
        property_mock(
            request, _CatXCatUnweightedCubeCounts, "table_base", return_value=9
        )
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )

        assert unweighted_cube_counts.table_bases.tolist() == [[9, 9, 9], [9, 9, 9]]

    @pytest.mark.parametrize(
        ("raw_unweighted_counts", "expected"),
        (
            # --- (1, 3) array ---
            ([[1, 2, 3]], [6]),
            # --- (2, 3) array ---
            ([[1, 2, 3], [4, 5, 6]], [6, 15]),
            # --- (3, 1) array ---
            ([[1], [2], [3]], [1, 2, 3]),
        ),
    )
    def it_knows_its_rows_base(self, raw_unweighted_counts, expected):
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_base.tolist() == expected

    def it_knows_its_unweighted_counts(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.unweighted_counts.tolist() == [
            [1, 2, 3],
            [4, 5, 6],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_unweighted_counts(self, request):
        """(2, 3) np.int ndarray of unweighted cube-counts as received from Cube."""
        return np.array([[1, 2, 3], [4, 5, 6]])


class Describe_CatXMrUnweightedCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._CatXMrUnweightedCubeCounts`."""

    def it_knows_its_columns_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_base.tolist() == [5, 7, 9]

    def it_knows_its_columns_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_pruning_base.tolist() == [14, 14, 14]

    def it_knows_its_row_bases(self, request):
        property_mock(
            request,
            _CatXMrUnweightedCubeCounts,
            "rows_base",
            return_value=np.array([[1, 2, 3], [4, 5, 6]]),
        )
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(None, None, None)

        assert unweighted_cube_counts.row_bases.tolist() == [[1, 2, 3], [4, 5, 6]]

    def it_knows_its_rows_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_base.tolist() == [[7, 7, 7], [7, 7, 7]]

    def it_knows_its_rows_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_pruning_base.tolist() == [21, 21]

    def it_knows_its_table_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.table_base.tolist() == [14, 14, 14]

    def it_knows_its_table_bases(self, request, raw_unweighted_counts):
        property_mock(
            request,
            _CatXMrUnweightedCubeCounts,
            "table_base",
            return_value=np.array([5, 4, 3]),
        )
        property_mock(
            request,
            _CatXMrUnweightedCubeCounts,
            "unweighted_counts",
            return_value=np.array([[0, 0, 0], [0, 0, 0]]),
        )
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )

        assert unweighted_cube_counts.table_bases.tolist() == [[5, 4, 3], [5, 4, 3]]

    def it_knows_its_unweighted_counts(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.unweighted_counts.tolist() == [
            [1, 2, 3],
            [4, 5, 6],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_unweighted_counts(self):
        """(2, 3, 2) np.int ndarray of unweighted cube-counts as received from Cube."""
        return np.array(
            [  # -- axes are (rows, cols, sel/not) --
                # --sel/not--
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


class Describe_MrXCatUnweightedCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXCatUnweightedCubeCounts`."""

    def it_knows_its_columns_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_base.tolist() == [[5, 7, 9], [7, 12, 11]]

    def it_knows_its_columns_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_pruning_base.tolist() == [12, 19, 20]

    def it_knows_its_row_bases(self, request):
        property_mock(
            request,
            _MrXCatUnweightedCubeCounts,
            "rows_base",
            return_value=np.array([1, 2, 3]),
        )
        property_mock(
            request,
            _MrXCatUnweightedCubeCounts,
            "unweighted_counts",
            return_value=np.array([[0, 0], [0, 0], [0, 0]]),
        )
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(None, None, None)

        assert unweighted_cube_counts.row_bases.tolist() == [[1, 1], [2, 2], [3, 3]]

    def it_knows_its_rows_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_base.tolist() == [6, 24]

    def it_knows_its_rows_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_pruning_base.tolist() == [21, 30]

    def it_knows_its_table_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.table_base.tolist() == [21, 30]

    def it_knows_its_table_bases(self, request, raw_unweighted_counts):
        property_mock(
            request,
            _MrXCatUnweightedCubeCounts,
            "table_base",
            return_value=np.array([6, 5]),
        )
        property_mock(
            request,
            _MrXCatUnweightedCubeCounts,
            "unweighted_counts",
            return_value=np.array([[0, 0, 0], [0, 0, 0]]),
        )
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )

        assert unweighted_cube_counts.table_bases.tolist() == [[6, 6, 6], [5, 5, 5]]

    def it_knows_its_unweighted_counts(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXCatUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.unweighted_counts.tolist() == [
            [1, 2, 3],
            [7, 8, 9],
        ]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_unweighted_counts(self):
        """(2, 3, 2) np.int ndarray of unweighted cube-counts as received from Cube."""
        return np.array(
            # -- axes are (rows, sel/not, cols) --
            [
                # -- 0  1  2 -- cols ---
                [  # -- row 0 ----------
                    [1, 2, 3],  # -- sel
                    [4, 5, 6],  # -- not
                ],
                [  # -- row 1 ----------
                    [7, 8, 9],  # -- sel
                    [0, 4, 2],  # -- not
                ],
            ]
        )


class Describe_MrXMrUnweightedCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._MrXMrUnweightedCubeCounts`."""

    def it_knows_its_columns_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_base.tolist() == [[2, 4], [10, 12]]

    def it_knows_its_columns_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.columns_pruning_base.tolist() == [12, 16]

    def it_knows_its_row_bases(self, request):
        property_mock(
            request,
            _MrXMrUnweightedCubeCounts,
            "rows_base",
            return_value=np.array([[1, 2], [5, 6]]),
        )
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(None, None, None)

        assert unweighted_cube_counts.row_bases.tolist() == [[1, 2], [5, 6]]

    def it_knows_its_rows_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_base.tolist() == [[8, 8], [8, 8]]

    def it_knows_its_rows_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.rows_pruning_base.tolist() == [16, 16]

    def it_knows_its_table_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.table_base.tolist() == [[16, 16], [16, 16]]

    def it_knows_its_table_bases(self, request):
        property_mock(
            request,
            _MrXMrUnweightedCubeCounts,
            "table_base",
            return_value=np.array([[3, 2], [7, 6]]),
        )
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(None, None, None)

        assert unweighted_cube_counts.table_bases.tolist() == [[3, 2], [7, 6]]

    def it_knows_its_unweighted_counts(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrXMrUnweightedCubeCounts(
            None, raw_unweighted_counts, None
        )
        assert unweighted_cube_counts.unweighted_counts.tolist() == [[0, 1], [4, 5]]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_unweighted_counts(self):
        """(2, 2, 2, 2) np.int ndarray of unweighted cube-counts as from Cube."""
        return np.array(
            # -- axes are (rows, sel/not, cols, sel/not) --
            [
                [  # -- row 0 -------------
                    # --sel/not--
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


# === LEGACY CUBE-RESULT MATRIX TESTS (should go away after measure consolidation) ===


class DescribeBaseCubeResultMatrix(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure.BaseCubeResultMatrix` object."""

    @pytest.mark.parametrize(
        "dimension_types, MatrixCls",
        (
            ((DT.MR, DT.MR), _MrXMrMatrix),
            ((DT.MR, DT.CAT), _MrXCatMatrix),
            ((DT.CAT, DT.MR), _CatXMrMatrix),
            ((DT.CAT, DT.CAT), _CatXCatMatrix),
        ),
    )
    def it_provides_a_factory_for_constructing_a_matrix_objects(
        self, request, dimension_types, MatrixCls
    ):
        cube_ = instance_mock(request, Cube)
        dimensions_ = (
            instance_mock(request, Dimension),
            instance_mock(request, Dimension),
        )
        cube_.dimension_types = dimension_types
        MatrixCls_ = class_mock(
            request, "cr.cube.matrix.cubemeasure.%s" % MatrixCls.__name__
        )

        _sliced_counts = method_mock(
            request,
            BaseCubeResultMatrix,
            "_sliced_counts",
            return_value=([[1], [2]], [[3], [4]]),
        )

        matrix = BaseCubeResultMatrix.factory(cube_, dimensions_, slice_idx=17)

        _sliced_counts.assert_called_once_with(cube_, 17)
        MatrixCls_.assert_called_once_with(dimensions_, [[1], [2]], [[3], [4]])
        assert matrix is MatrixCls_.return_value

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
    def it_knows_its_cube_slice_expression_to_help(
        self, cube_, slice_idx, dim_types, expected
    ):
        cube_.dimension_types = dim_types
        cube_.ndim = len(dim_types)

        s = BaseCubeResultMatrix._cube_slice_expression(cube_, slice_idx)

        assert s == expected

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
        _cube_slice_expression_ = method_mock(
            request,
            BaseCubeResultMatrix,
            "_cube_slice_expression",
            return_value=counts_slice,
        )

        sliced_counts = BaseCubeResultMatrix._sliced_counts(cube_, slice_idx=23)

        _cube_slice_expression_.assert_called_once_with(cube_, 23)
        counts, unweighted, with_missing = sliced_counts
        assert counts.tolist() == expected
        assert unweighted.tolist() == expected

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)


class Describe_CatXCatMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXCatMatrix` object."""

    def it_knows_its_table_base(self):
        unweighted_counts = np.array([[1, 2, 3], [4, 5, 6]])
        assert _CatXCatMatrix(None, None, unweighted_counts).table_base == 21

    def it_knows_its_unweighted_counts(self):
        unweighted_cube_counts = np.array([[1, 2, 3], [4, 5, 6]])
        matrix = _CatXCatMatrix(None, None, unweighted_cube_counts)

        assert matrix.unweighted_counts.tolist() == [[1, 2, 3], [4, 5, 6]]


class Describe_CatXMrMatrix(object):
    """Unit test suite for `cr.cube.matrix._CatXMrMatrix` object."""

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


class Describe_MrXCatMatrix(object):
    """Unit test suite for `cr.cube.matrix._MrXCatMatrix` object."""

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


class Describe_MrXMrMatrix(object):
    """Unit test suite for `cr.cube.matrix._MrXMrMatrix` object."""

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
