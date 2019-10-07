# encoding: utf-8

"""Unit test suite for `cr.cube.cube.Cubeset` class."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

from cr.cube.cube import CubeSet, Cube
from cr.cube.cubepart import _Slice, _Strand, _Nub
from cr.cube.enum import DIMENSION_TYPE as DT

from ..unitutil import instance_mock, property_mock, method_mock


class DescribeCubeSet(object):
    def it_provides_access_to_its_Cube_objects_to_help(self, _iter_cubes_, cube_):
        _iter_cubes_.return_value = (c for c in (cube_, cube_, cube_))
        cube_set = CubeSet(None, None, None, None)

        cubes = cube_set._cubes

        assert cubes == (cube_, cube_, cube_)

    def it_knows_whether_it_can_show_pairwise(
        self, request, can_show_pairwise_fixture, _cubes_prop_
    ):
        cubes_dimtypes, expected_value = can_show_pairwise_fixture
        _cubes_prop_.return_value = tuple(
            instance_mock(
                request, Cube, dimension_types=cube_dimtypes, ndim=len(cube_dimtypes)
            )
            for cube_dimtypes in cubes_dimtypes
        )
        cube_set = CubeSet(None, None, None, None)

        can_show_pairwise = cube_set.can_show_pairwise

        assert can_show_pairwise is expected_value

    def it_knows_its_name(self, _cubes_prop_, cube_):
        cube_.name = "Beverage"
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        name = cube_set.name

        assert name == "Beverage"

    def it_knows_its_description(self, _cubes_prop_, cube_):
        cube_.description = "Are you male or female?"
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        description = cube_set.description

        assert description == "Are you male or female?"

    def it_knows_its_missing_count(self, missing_count_fixture, _cubes_prop_, cube_):
        first_cube_missing_count, expected_value = missing_count_fixture
        cube_.missing = first_cube_missing_count
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        missing_count = cube_set.missing_count

        assert missing_count == expected_value

    def it_knows_its_sequence_of_cube_objects(self, cube_, _cubedict_):
        cubes_ = (cube_,) * 3
        _cubedict_.return_value = cube_._cube_dict
        transforms_ = [{}, {}, {}]
        cube_set = CubeSet(cubes_, transforms_, None, None)

        cube_sequence = tuple(cube_set._iter_cubes())

        assert len(cube_sequence) == 3
        assert all(isinstance(cube, Cube) for cube in cube_sequence)

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

        has_weighted_counts = cube_set.has_weighted_counts

        assert has_weighted_counts == expected_value

    def it_knows_when_it_is_ca_as_0th(self, is_ca_as_0th_fixture, _cubes_prop_, cube_):
        ncubes, expected_value = is_ca_as_0th_fixture
        cubes_ = (cube_,) * ncubes
        cubes_[0].dimension_types = (DT.CA_SUBVAR,) * ncubes
        _cubes_prop_.return_value = cubes_
        cube_set = CubeSet(cubes_, None, None, None)

        is_ca_as_0th = cube_set.is_ca_as_0th

        assert is_ca_as_0th == expected_value

    def it_provides_access_to_the_partition_sets(
        self, partition_set_fixture, _cubes_prop_, cube_
    ):
        cube_partition, expected_value = partition_set_fixture
        cube_.partitions = cube_partition
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        partition_sets = cube_set.partition_sets

        assert partition_sets == expected_value

    def it_has_proper_population_fraction(
        self, population_fraction_fixture, cube_, _cubes_prop_
    ):
        population_fraction, expected_value = population_fraction_fixture
        cube_.population_fraction = population_fraction
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        cubeset_population_fraction = cube_set.population_fraction

        np.testing.assert_almost_equal(cubeset_population_fraction, expected_value)

    # fixtures ---------------------------------------------

    @pytest.fixture(
        params=[
            ((), False),
            (((DT.CAT, DT.CAT),), False),
            (((DT.TEXT,), (DT.TEXT,)), False),
            (((DT.CAT,), (DT.CAT, DT.CAT)), True),
            (((DT.TEXT,), (DT.CAT, DT.CAT)), True),
            (((DT.MR_CAT,), (DT.CAT, DT.CAT)), True),
            (((DT.CAT,), (DT.CAT, DT.MR_CAT)), False),
            (((DT.CAT, DT.CAT), (DT.CAT, DT.MR_CAT)), False),
            (((DT.DATETIME,), (DT.DATETIME, DT.DATETIME)), True),
            (((DT.DATETIME, DT.DATETIME), (DT.DATETIME, DT.DATETIME)), True),
            (((DT.DATETIME, DT.DATETIME), (DT.DATETIME, DT.MR_CAT)), False),
        ]
    )
    def can_show_pairwise_fixture(self, request):
        cubes_dimtypes, expected_value = request.param
        return cubes_dimtypes, expected_value

    @pytest.fixture(params=[(True, True), (False, False)])
    def has_means_fixture(self, request):
        first_cube_has_means, expected_value = request.param
        return first_cube_has_means, expected_value

    @pytest.fixture(params=[(True, True), (False, False)])
    def has_weighted_counts_fixture(self, request):
        first_cube_has_w_counts, expected_value = request.param
        return first_cube_has_w_counts, expected_value

    @pytest.fixture(params=[(2, True), (1, False)])
    def is_ca_as_0th_fixture(self, request):
        ncubes, expected_value = request.param
        return ncubes, expected_value

    @pytest.fixture(params=[(34, 34), (0, 0)])
    def missing_count_fixture(self, request):
        first_cube_missing_count, expected_value = request.param
        return first_cube_missing_count, expected_value

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
        cube_partitions, expected_value = request.param
        return cube_partitions, expected_value

    @pytest.fixture(params=[(1.0, 1.0), (0.54, 0.54), (np.nan, np.nan)])
    def population_fraction_fixture(self, request):
        population_fraction, expected_value = request.param
        return population_fraction, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def _cubedict_(self, request):
        return property_mock(request, Cube, "_cube_dict")

    @pytest.fixture
    def _cubes_prop_(self, request):
        return property_mock(request, CubeSet, "_cubes")

    @pytest.fixture
    def _iter_cubes_(self, request):
        return method_mock(request, CubeSet, "_iter_cubes")
