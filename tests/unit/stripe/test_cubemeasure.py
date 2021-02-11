# encoding: utf-8

"""Unit test suite for `cr.cube.stripe.cubemeasure` module."""

import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import Dimension
from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.stripe.cubemeasure import (
    _BaseUnweightedCubeCounts,
    _CatUnweightedCubeCounts,
    CubeMeasures,
    _MrUnweightedCubeCounts,
)

from ...unitutil import class_mock, instance_mock


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
