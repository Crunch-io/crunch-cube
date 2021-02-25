# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.cubemeasure` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.stripe.cubemeasure import (
    _BaseUnweightedCubeCounts,
    _BaseWeightedCubeCounts,
    _CatUnweightedCubeCounts,
    _CatWeightedCubeCounts,
    CubeMeasures,
    _MrUnweightedCubeCounts,
    _MrWeightedCubeCounts,
)

from ...unitutil import class_mock, instance_mock, property_mock


class DescribeCubeMeasures(object):
    """Unit-test suite for `cr.cube.stripe.cubemeasure.CubeMeasures` object."""

    def it_provides_access_to_the_unweighted_cube_counts_object(
        self, request, cube_, rows_dimension_
    ):
        unweighted_cube_counts_ = instance_mock(request, _BaseUnweightedCubeCounts)
        _BaseUnweightedCubeCounts_ = class_mock(
            request, "cr.cube.stripe.cubemeasure._BaseUnweightedCubeCounts"
        )
        _BaseUnweightedCubeCounts_.factory.return_value = unweighted_cube_counts_
        cube_measures = CubeMeasures(cube_, rows_dimension_, False, slice_idx=7)

        unweighted_cube_counts = cube_measures.unweighted_cube_counts

        _BaseUnweightedCubeCounts_.factory.assert_called_once_with(
            cube_, rows_dimension_, False, 7
        )
        assert unweighted_cube_counts is unweighted_cube_counts_

    def it_provides_access_to_the_weighted_cube_counts_object(
        self, request, cube_, rows_dimension_
    ):
        weighted_cube_counts_ = instance_mock(request, _BaseWeightedCubeCounts)
        _BaseWeightedCubeCounts_ = class_mock(
            request, "cr.cube.stripe.cubemeasure._BaseWeightedCubeCounts"
        )
        _BaseWeightedCubeCounts_.factory.return_value = weighted_cube_counts_
        cube_measures = CubeMeasures(cube_, rows_dimension_, False, slice_idx=7)

        weighted_cube_counts = cube_measures.weighted_cube_counts

        _BaseWeightedCubeCounts_.factory.assert_called_once_with(
            cube_, rows_dimension_, False, 7
        )
        assert weighted_cube_counts is weighted_cube_counts_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def rows_dimension_(self, request):
        return instance_mock(request, Dimension)


# === UNWEIGHTED COUNTS ===


class Describe_BaseUnweightedCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseUnweightedCubeCounts`."""

    @pytest.mark.parametrize(
        "ca_as_0th, rows_dimension_type, UnweightedCubeCountsCls, unweighted_counts",
        (
            (True, DT.CA_CAT, _CatUnweightedCubeCounts, [[4, 5, 6], [1, 2, 3]]),
            (False, DT.MR, _MrUnweightedCubeCounts, [1, 2, 3]),
            (False, DT.CAT, _CatUnweightedCubeCounts, [1, 2, 3]),
        ),
    )
    def it_provides_a_factory_for_constructing_unweighted_cube_count_objects(
        self,
        request,
        ca_as_0th,
        rows_dimension_type,
        UnweightedCubeCountsCls,
        unweighted_counts,
    ):
        cube_ = instance_mock(request, Cube, unweighted_counts=unweighted_counts)
        rows_dimension_ = instance_mock(
            request, Dimension, dimension_type=rows_dimension_type
        )
        unweighted_cube_counts_ = instance_mock(request, UnweightedCubeCountsCls)
        UnweightedCubeCountsCls_ = class_mock(
            request,
            "cr.cube.stripe.cubemeasure.%s" % UnweightedCubeCountsCls.__name__,
            return_value=unweighted_cube_counts_,
        )

        unweighted_cube_counts = _BaseUnweightedCubeCounts.factory(
            cube_, rows_dimension_, ca_as_0th, slice_idx=1
        )

        UnweightedCubeCountsCls_.assert_called_once_with(rows_dimension_, [1, 2, 3])
        assert unweighted_cube_counts is unweighted_cube_counts_


class Describe_CatUnweightedCubeCounts(object):
    """Unit-test suite for `cr.cube.stripe.cubemeasure._CatUnweightedCubeCounts`."""

    def it_knows_its_bases(self, request, raw_unweighted_counts):
        property_mock(
            request,
            _CatUnweightedCubeCounts,
            "table_base",
            return_value=42,
        )
        unweighted_cube_counts = _CatUnweightedCubeCounts(None, raw_unweighted_counts)
        assert unweighted_cube_counts.bases.tolist() == [42, 42, 42]

    def it_knows_its_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatUnweightedCubeCounts(None, raw_unweighted_counts)
        assert unweighted_cube_counts.pruning_base.tolist() == [1, 2, 3]

    def it_knows_its_table_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatUnweightedCubeCounts(None, raw_unweighted_counts)
        assert unweighted_cube_counts.table_base == 6

    def it_knows_its_unweighted_counts(self, raw_unweighted_counts):
        unweighted_cube_counts = _CatUnweightedCubeCounts(None, raw_unweighted_counts)
        assert unweighted_cube_counts.unweighted_counts.tolist() == [1, 2, 3]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_unweighted_counts(self, request):
        """(3,) np.int ndarray of unweighted cube-counts as received from Cube."""
        return np.array([1, 2, 3])


class Describe_MrUnweightedCubeCounts(object):
    """Unit-test suite for `cr.cube.stripe.cubemeasure._MrUnweightedCubeCounts`."""

    def it_knows_its_bases(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrUnweightedCubeCounts(None, raw_unweighted_counts)
        assert unweighted_cube_counts.bases.tolist() == [3, 7, 11]

    def it_knows_its_pruning_base(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrUnweightedCubeCounts(None, raw_unweighted_counts)
        assert unweighted_cube_counts.pruning_base.tolist() == [3, 7, 11]

    def it_knows_its_unweighted_counts(self, raw_unweighted_counts):
        unweighted_cube_counts = _MrUnweightedCubeCounts(None, raw_unweighted_counts)
        assert unweighted_cube_counts.unweighted_counts.tolist() == [1, 3, 5]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_unweighted_counts(self, request):
        """(3, 2) np.int ndarray of unweighted cube-counts as received from Cube."""
        return np.array([[1, 2], [3, 4], [5, 6]])


# === WEIGHTED COUNTS ===


class Describe_BaseWeightedCubeCounts(object):
    """Unit test suite for `cr.cube.matrix.cubemeasure._BaseWeightedCubeCounts`."""

    @pytest.mark.parametrize(
        "ca_as_0th, rows_dimension_type, WeightedCubeCountsCls, weighted_counts",
        (
            (
                True,
                DT.CA_CAT,
                _CatWeightedCubeCounts,
                [[4.4, 5.5, 6.6], [1.1, 2.2, 3.3]],
            ),
            (False, DT.MR, _MrWeightedCubeCounts, [1.1, 2.2, 3.3]),
            (False, DT.CAT, _CatWeightedCubeCounts, [1.1, 2.2, 3.3]),
        ),
    )
    def it_provides_a_factory_for_constructing_weighted_cube_count_objects(
        self,
        request,
        ca_as_0th,
        rows_dimension_type,
        WeightedCubeCountsCls,
        weighted_counts,
    ):
        cube_ = instance_mock(request, Cube, counts=weighted_counts)
        rows_dimension_ = instance_mock(
            request, Dimension, dimension_type=rows_dimension_type
        )
        weighted_cube_counts_ = instance_mock(request, WeightedCubeCountsCls)
        WeightedCubeCountsCls_ = class_mock(
            request,
            "cr.cube.stripe.cubemeasure.%s" % WeightedCubeCountsCls.__name__,
            return_value=weighted_cube_counts_,
        )

        weighted_cube_counts = _BaseWeightedCubeCounts.factory(
            cube_, rows_dimension_, ca_as_0th, slice_idx=1
        )

        WeightedCubeCountsCls_.assert_called_once_with(rows_dimension_, [1.1, 2.2, 3.3])
        assert weighted_cube_counts is weighted_cube_counts_


class Describe_CatWeightedCubeCounts(object):
    """Unit-test suite for `cr.cube.stripe.cubemeasure._CatWeightedCubeCounts`."""

    def it_knows_its_bases(self, request, raw_weighted_counts):
        property_mock(
            request,
            _CatWeightedCubeCounts,
            "table_margin",
            return_value=42.42,
        )
        weighted_cube_counts = _CatWeightedCubeCounts(None, raw_weighted_counts)
        assert weighted_cube_counts.bases.tolist() == [42.42, 42.42, 42.42]

    def it_knows_its_weighted_counts(self, raw_weighted_counts):
        weighted_cube_counts = _CatWeightedCubeCounts(None, raw_weighted_counts)
        assert weighted_cube_counts.weighted_counts.tolist() == [1.1, 2.2, 3.3]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_weighted_counts(self, request):
        """(3,) np.int ndarray of weighted cube-counts as received from Cube."""
        return np.array([1.1, 2.2, 3.3])


class Describe_MrWeightedCubeCounts(object):
    """Unit-test suite for `cr.cube.stripe.cubemeasure._MrWeightedCubeCounts`."""

    def it_knows_its_weighted_counts(self, raw_weighted_counts):
        weighted_cube_counts = _MrWeightedCubeCounts(None, raw_weighted_counts)
        assert weighted_cube_counts.weighted_counts.tolist() == [1.1, 3.3, 5.5]

    # fixtures -------------------------------------------------------

    @pytest.fixture
    def raw_weighted_counts(self, request):
        """(3, 2) np.int ndarray of weighted cube-counts as received from Cube."""
        return np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
