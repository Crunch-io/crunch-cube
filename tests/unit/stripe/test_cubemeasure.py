# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.cubemeasure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.stripe.cubemeasure import (
    _BaseCubeMeans,
    _BaseCubeSums,
    _BaseCubeCounts,
    _CatCubeCounts,
    _CatCubeMeans,
    _CatCubeSums,
    CubeMeasures,
    _MrCubeCounts,
    _MrCubeMeans,
    _MrCubeSums,
    _NumArrCubeCounts,
)

from ...unitutil import class_mock, instance_mock, property_mock


class DescribeCubeMeasures:
    """Unit-test suite for `cr.cube.stripe.cubemeasure.CubeMeasures` object."""

    def it_provides_access_to_the_cube_means_object(
        self, request, cube_, rows_dimension_
    ):
        cube_means_ = instance_mock(request, _BaseCubeMeans)
        _BaseCubeMeans_ = class_mock(
            request, "cr.cube.stripe.cubemeasure._BaseCubeMeans"
        )
        _BaseCubeMeans_.factory.return_value = cube_means_
        cube_measures = CubeMeasures(cube_, rows_dimension_, None, None)

        cube_means = cube_measures.cube_means

        _BaseCubeMeans_.factory.assert_called_once_with(cube_, rows_dimension_)
        assert cube_means is cube_means_

    def it_provides_access_to_the_cube_sum_object(
        self, request, cube_, rows_dimension_
    ):
        cube_sum_ = instance_mock(request, _BaseCubeSums)
        _BaseCubeSums_ = class_mock(request, "cr.cube.stripe.cubemeasure._BaseCubeSums")
        _BaseCubeSums_.factory.return_value = cube_sum_
        cube_measures = CubeMeasures(cube_, rows_dimension_, None, None)

        cube_sum = cube_measures.cube_sum

        _BaseCubeSums_.factory.assert_called_once_with(cube_, rows_dimension_)
        assert cube_sum is cube_sum_

    @pytest.mark.parametrize(
        "unweighted_counts, unweighted_valid_counts, counts_used",
        (
            ("counts", None, "counts"),
            ("counts", "valid", "valid"),
        ),
    )
    def it_provides_access_to_the_unweighted_cube_counts_object(
        self,
        cube_,
        cube_counts_,
        _BaseCubeCounts_,
        rows_dimension_,
        unweighted_counts,
        unweighted_valid_counts,
        counts_used,
    ):
        cube_.unweighted_counts = unweighted_counts
        cube_.unweighted_valid_counts = unweighted_valid_counts
        _BaseCubeCounts_.factory.return_value = cube_counts_
        cube_measures = CubeMeasures(cube_, rows_dimension_, False, slice_idx=7)

        unweighted_cube_counts = cube_measures.unweighted_cube_counts

        _BaseCubeCounts_.factory.assert_called_once_with(
            counts_used, rows_dimension_, False, 7
        )
        assert unweighted_cube_counts is cube_counts_

    @pytest.mark.parametrize(
        "weighted_counts, weighted_valid_counts, counts_used",
        (
            ("counts", None, "counts"),
            ("counts", "valid", "valid"),
        ),
    )
    def it_provides_access_to_the_weighted_cube_counts_object(
        self,
        cube_,
        cube_counts_,
        _BaseCubeCounts_,
        rows_dimension_,
        weighted_counts,
        weighted_valid_counts,
        counts_used,
    ):
        cube_.counts = weighted_counts
        cube_.weighted_valid_counts = weighted_valid_counts
        _BaseCubeCounts_.factory.return_value = cube_counts_
        cube_measures = CubeMeasures(cube_, rows_dimension_, False, slice_idx=7)

        weighted_cube_counts = cube_measures.weighted_cube_counts

        _BaseCubeCounts_.factory.assert_called_once_with(
            counts_used, rows_dimension_, False, 7
        )
        assert weighted_cube_counts is cube_counts_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _BaseCubeCounts_(self, request):
        return class_mock(request, "cr.cube.stripe.cubemeasure._BaseCubeCounts")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def cube_counts_(self, request):
        return instance_mock(request, _BaseCubeCounts)

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)


# === COUNTS ===


class Describe_BaseCubeCounts:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseUnweightedCubeCounts`."""

    @pytest.mark.parametrize(
        (
            "ca_as_0th",
            "rows_dimension_type",
            "CubeCountsCls",
            "counts",
            "expected_counts",
        ),
        (
            (
                True,
                DT.CA_CAT,
                _CatCubeCounts,
                [[4, 5, 6], [1, 2, 3]],
                [1, 2, 3],
            ),
            (False, DT.MR, _MrCubeCounts, [1, 2, 3], [1, 2, 3]),
            (False, DT.CAT, _CatCubeCounts, [1, 2, 3], [1, 2, 3]),
            (
                False,
                DT.NUM_ARRAY,
                _NumArrCubeCounts,
                [4, 5, 6],
                [4, 5, 6],
            ),
        ),
    )
    def it_provides_a_factory_for_constructing_unweighted_cube_count_objects(
        self,
        request,
        ca_as_0th,
        rows_dimension_type,
        CubeCountsCls,
        counts,
        expected_counts,
    ):
        rows_dimension_ = instance_mock(
            request, Dimension, dimension_type=rows_dimension_type
        )
        cube_counts_ = instance_mock(request, CubeCountsCls)
        CubeCountsCls_ = class_mock(
            request,
            "cr.cube.stripe.cubemeasure.%s" % CubeCountsCls.__name__,
            return_value=cube_counts_,
        )

        unweighted_cube_counts = _BaseCubeCounts.factory(
            counts, rows_dimension_, ca_as_0th, slice_idx=1
        )

        CubeCountsCls_.assert_called_once_with(rows_dimension_, expected_counts)
        assert unweighted_cube_counts is cube_counts_


class Describe_CatCubeCounts:
    """Unit-test suite for `cr.cube.stripe.cubemeasure._CatCubeCounts`."""

    def it_knows_its_bases(self, request, raw_counts):
        property_mock(
            request,
            _CatCubeCounts,
            "table_base",
            return_value=42,
        )
        cube_counts = _CatCubeCounts(None, raw_counts)
        assert cube_counts.bases.tolist() == [42, 42, 42]

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _CatCubeCounts(None, raw_counts)
        assert cube_counts.counts.tolist() == [1, 2, 3]

    def it_knows_its_pruning_base(self, raw_counts):
        cube_counts = _CatCubeCounts(None, raw_counts)
        assert cube_counts.pruning_base.tolist() == [1, 2, 3]

    def it_knows_its_table_base(self, raw_counts):
        cube_counts = _CatCubeCounts(None, raw_counts)
        assert cube_counts.table_base == 6

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(3,) np.int ndarray of cube-counts as received from Cube."""
        return np.array([1, 2, 3])


class Describe_MrCubeCounts:
    """Unit-test suite for `cr.cube.stripe.cubemeasure._MrCubeCounts`."""

    def it_knows_its_bases(self, raw_counts):
        cube_counts = _MrCubeCounts(None, raw_counts)
        assert cube_counts.bases.tolist() == [3, 7, 11]

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _MrCubeCounts(None, raw_counts)
        assert cube_counts.counts.tolist() == [1, 3, 5]

    def it_knows_its_pruning_base(self, raw_counts):
        cube_counts = _MrCubeCounts(None, raw_counts)
        assert cube_counts.pruning_base.tolist() == [3, 7, 11]

    def it_knows_its_table_base_is_None(self, raw_counts):
        cube_counts = _MrCubeCounts(None, raw_counts)
        assert cube_counts.table_base is None

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(3, 2) np.int ndarray of unweighted cube-counts as received from Cube."""
        return np.array([[1, 2], [3, 4], [5, 6]])


class Describe_NumArrCubeCounts:
    """Unit-test suite for `cr.cube.stripe.cubemeasure._NumArrCubeCounts`."""

    def it_knows_its_bases(self, raw_counts):
        cube_counts = _NumArrCubeCounts(None, raw_counts)
        assert cube_counts.bases.tolist() == [1, 2, 3]

    def it_knows_its_counts(self, raw_counts):
        cube_counts = _NumArrCubeCounts(None, raw_counts)
        assert cube_counts.counts.tolist() == [1, 2, 3]

    def it_knows_its_pruning_base(self, raw_counts):
        cube_counts = _NumArrCubeCounts(None, raw_counts)
        assert cube_counts.pruning_base.tolist() == [1, 2, 3]

    def it_knows_its_table_base_is_None(self, raw_counts):
        cube_counts = _NumArrCubeCounts(None, raw_counts)
        assert cube_counts.table_base is None

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_counts(self):
        """(3,) np.int ndarray of valid cube-counts as received from Cube."""
        return np.array([1, 2, 3])


# === MEANS ===


class Describe_BaseCubeMeans:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseCubeMeans`."""

    @pytest.mark.parametrize(
        "rows_dimension_type, CubeMeansCls, means",
        (
            (DT.CAT, _CatCubeMeans, [1, 2, 3]),
            (DT.MR, _MrCubeMeans, [[1, 6], [2, 5], [3, 4]]),
        ),
    )
    def it_provides_a_factory_for_constructing_cube_means_objects(
        self, request, rows_dimension_type, CubeMeansCls, means
    ):
        cube_ = instance_mock(request, Cube, means=means)
        rows_dimension_ = instance_mock(
            request, Dimension, dimension_type=rows_dimension_type
        )
        cube_means_ = instance_mock(request, CubeMeansCls)
        CubeMeansCls_ = class_mock(
            request,
            "cr.cube.stripe.cubemeasure.%s" % CubeMeansCls.__name__,
            return_value=cube_means_,
        )

        cube_means = _BaseCubeMeans.factory(cube_, rows_dimension_)

        CubeMeansCls_.assert_called_once_with(rows_dimension_, means)
        assert cube_means is cube_means_


class Describe_CatCubeMeans:
    """Unit-test suite for `cr.cube.stripe.cubemeasure._CatCubeMeans`."""

    def it_knows_its_means(self):
        cube_means = _CatCubeMeans(None, np.array([1.1, 2.2, 3.3]))
        assert cube_means.means == pytest.approx([1.1, 2.2, 3.3])

    def but_it_raises_value_error_when_the_cube_result_does_not_contain_means(
        self, request
    ):
        cube_ = instance_mock(request, Cube)
        cube_.means = None
        with pytest.raises(ValueError) as e:
            _CatCubeMeans(None, None).factory(cube_, None)

        assert str(e.value) == "cube-result does not contain cube-means measure"


class Describe_MrCubeMeans:
    """Unit-test suite for `cr.cube.stripe.cubemeasure._MrCubeMeans`."""

    def it_knows_its_means(self):
        cube_means = _MrCubeMeans(None, np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))
        assert cube_means.means == pytest.approx([1.1, 3.3, 5.5])

    def but_it_raises_value_error_when_the_cube_result_does_not_contain_means(
        self, request
    ):
        cube_ = instance_mock(request, Cube)
        cube_.means = None
        with pytest.raises(ValueError) as e:
            _CatCubeMeans(None, None).factory(cube_, None)

        assert str(e.value) == "cube-result does not contain cube-means measure"


# === SUM ===


class Describe_BaseCubeSums:
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseCubeSums`."""

    @pytest.mark.parametrize(
        "rows_dimension_type, CubeSumCls, sums",
        (
            (DT.CAT, _CatCubeSums, [1, 2, 3]),
            (DT.MR, _MrCubeSums, [[1, 6], [2, 5], [3, 4]]),
        ),
    )
    def it_provides_a_factory_for_constructing_cube_sums_objects(
        self, request, rows_dimension_type, CubeSumCls, sums
    ):
        cube_ = instance_mock(request, Cube, sums=sums)
        rows_dimension_ = instance_mock(
            request, Dimension, dimension_type=rows_dimension_type
        )
        cube_sums_ = instance_mock(request, CubeSumCls)
        CubeSumCls_ = class_mock(
            request,
            "cr.cube.stripe.cubemeasure.%s" % CubeSumCls.__name__,
            return_value=cube_sums_,
        )

        cube_sums = _BaseCubeSums.factory(cube_, rows_dimension_)

        CubeSumCls_.assert_called_once_with(rows_dimension_, sums)
        assert cube_sums is cube_sums_


class Describe_CatCubeSums:
    """Unit-test suite for `cr.cube.stripe.cubemeasure._CatCubeSum`."""

    def it_knows_its_sum(self):
        cube_sum = _CatCubeSums(None, np.array([1, 2, 3]))
        assert cube_sum.sums == pytest.approx([1, 2, 3])

    def but_it_raises_value_error_when_the_cube_result_does_not_contain_sums(
        self, request
    ):
        cube_ = instance_mock(request, Cube)
        cube_.sums = None
        with pytest.raises(ValueError) as e:
            _CatCubeSums(None, None).factory(cube_, None)

        assert str(e.value) == "cube-result does not contain cube-sum measure"


class Describe_MrCubeSums:
    """Unit-test suite for `cr.cube.stripe.cubemeasure._MrCubeSum`."""

    def it_knows_its_sum(self):
        cube_sum = _MrCubeSums(None, np.array([[1, 2], [3, 4], [5, 6]]))
        assert cube_sum.sums == pytest.approx([1, 3, 5])

    def but_it_raises_value_error_when_the_cube_result_does_not_contain_sum(
        self, request
    ):
        cube_ = instance_mock(request, Cube)
        cube_.sums = None
        with pytest.raises(ValueError) as e:
            _CatCubeSums(None, None).factory(cube_, None)

        assert str(e.value) == "cube-result does not contain cube-sum measure"
