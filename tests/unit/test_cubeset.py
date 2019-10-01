# encoding: utf-8

"""Unit test suite for `cr.cube.cube.Cubeset` class."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

from cr.cube.cube import CubeSet, Cube
from cr.cube.cubepart import _Slice, _Strand, _Nub
from cr.cube.enum import DIMENSION_TYPE as DT

from ..unitutil import instance_mock, property_mock


class DescribeCrunchCubeSet(object):
    def it_knows_whether_it_has_means(self, has_means_fixture, _cubes_prop_, cube_):
        first_cube_has_means, expected_value = has_means_fixture
        cube_.has_means = first_cube_has_means
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        has_means = cube_set.has_means

        assert has_means == expected_value

    def it_knows_whether_it_has_weighted_counts(
        self, has_weighted_counts_fixture, _cubes_prop_, cube_
    ):
        first_cube_has_w_counts, expected_value = has_weighted_counts_fixture
        cube_.is_weighted = first_cube_has_w_counts
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        is_weighted = cube_set.has_weighted_counts

        assert is_weighted == expected_value

    def it_knows_its_name(self, has_name_fixture, _cubes_prop_, cube_):
        first_cube_name, expected_value = has_name_fixture
        cube_.name = first_cube_name
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        name = cube_set.name

        assert name == expected_value

    def it_knows_its_description(self, has_description_fixture, _cubes_prop_, cube_):
        first_cube_description, expected_value = has_description_fixture
        cube_.description = first_cube_description
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        description = cube_set.description

        assert description == expected_value

    def it_can_show_pairwise(self, can_show_pairwise_fixture, _cubes_prop_, cube_):
        ndim, expected_value = can_show_pairwise_fixture
        cube_.ndim = ndim
        _cubes_prop_.return_value = ndim * (cube_,)
        cube_set = CubeSet(None, None, None, None)

        can_show_pairwise = cube_set.can_show_pairwise

        assert can_show_pairwise == expected_value

    def it_knows_its_missing_count(self, missing_count_fixture, _cubes_prop_, cube_):
        first_cube_missing_count, expected_value = missing_count_fixture
        cube_.missing = first_cube_missing_count
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        missing_count = cube_set.missing_count

        assert missing_count == expected_value

    def it_knows_when_it_is_ca_as_0th(self, is_ca_as_0th_fixture, _cubes_prop_, cube_):
        ncubes, expected_value = is_ca_as_0th_fixture
        cubes_ = (cube_,) * ncubes
        cubes_[0].dimension_types = (DT.CA_SUBVAR,) * ncubes
        _cubes_prop_.return_value = cubes_
        cube_set = CubeSet(cubes_, None, None, None)

        isca_as_0th = cube_set.is_ca_as_0th

        assert isca_as_0th == expected_value

    def it_has_the_right_partition_set(
        self, partition_set_fixture, _cubes_prop_, cube_
    ):
        partition_set, expected_value = partition_set_fixture
        cube_.partitions = partition_set
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        partitions_set = cube_set.partition_sets

        assert partitions_set == expected_value

    def it_has_proper_population_fraction(
        self, population_fraction_fixture, cube_, _cubes_prop_
    ):
        population_fraction, expected_value = population_fraction_fixture
        cube_.population_fraction = population_fraction
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        cubeset_population_fraction = cube_set.population_fraction

        np.testing.assert_almost_equal(cubeset_population_fraction, expected_value)

    # fixture components ---------------------------------------------

    @pytest.fixture(params=[(1.0, 1.0), (0.54, 0.54), (np.nan, np.nan)])
    def population_fraction_fixture(self, request):
        population_fraction, expected_value = request.param
        return population_fraction, expected_value

    @pytest.fixture(
        # 3D, 2D, 1D, Nub
        params=[
            ((_Strand, _Slice, _Slice), ((_Strand,), (_Slice,), (_Slice,))),
            ((_Slice, _Slice), ((_Slice,), (_Slice,))),
            ((_Slice,), ((_Slice,),)),
            ((_Nub,), ((_Nub,),)),
        ]
    )
    def partition_set_fixture(self, request):
        partition_set, expected_value = request.param
        return partition_set, expected_value

    @pytest.fixture(params=[(2, True), (1, False)])
    def is_ca_as_0th_fixture(self, request):
        ncubes, expected_value = request.param
        return ncubes, expected_value

    @pytest.fixture(params=[(True, True), (False, False)])
    def has_means_fixture(self, request):
        first_cube_has_means, expected_value = request.param
        return first_cube_has_means, expected_value

    @pytest.fixture(params=[(True, True), (False, False)])
    def has_weighted_counts_fixture(self, request):
        first_cube_has_w_counts, expected_value = request.param
        return first_cube_has_w_counts, expected_value

    @pytest.fixture(params=[("MyCube", "MyCube"), (None, None)])
    def has_name_fixture(self, request):
        first_cube_name, expected_value = request.param
        return first_cube_name, expected_value

    @pytest.fixture(params=[("MyCube Description", "MyCube Description"), (None, None)])
    def has_description_fixture(self, request):
        first_cube_description, expected_value = request.param
        return first_cube_description, expected_value

    @pytest.fixture(params=[(3, True), (1, False)])
    def can_show_pairwise_fixture(self, request):
        ndim, expected_value = request.param
        return ndim, expected_value

    @pytest.fixture(params=[(34, 34), (0, 0)])
    def missing_count_fixture(self, request):
        first_cube_missing_count, expected_value = request.param
        return first_cube_missing_count, expected_value

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def _cubes_prop_(self, request):
        return property_mock(request, CubeSet, "_cubes")
