# encoding: utf-8

"""Unit test suite for `cr.cube.cube` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

from cr.cube.cube import Cube, CubeSet, _Measures
from cr.cube.cubepart import _Slice, _Strand, _Nub
from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.dimension import _ApparentDimensions, Dimension

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---
from ..unitutil import instance_mock, property_mock, method_mock


class DescribeCubeSet(object):
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

    def it_knows_its_description(self, _cubes_prop_, cube_):
        cube_.description = "Are you male or female?"
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        description = cube_set.description

        assert description == "Are you male or female?"

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

    def it_knows_its_missing_count(self, missing_count_fixture, _cubes_prop_, cube_):
        first_cube_missing_count, expected_value = missing_count_fixture
        cube_.missing = first_cube_missing_count
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        missing_count = cube_set.missing_count

        assert missing_count == expected_value

    def it_knows_its_name(self, _cubes_prop_, cube_):
        cube_.name = "Beverage"
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        name = cube_set.name

        assert name == "Beverage"

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

    def it_provides_access_to_its_Cube_objects_to_help(self, _iter_cubes_, cube_):
        _iter_cubes_.return_value = (c for c in (cube_, cube_, cube_))
        cube_set = CubeSet(None, None, None, None)

        cubes = cube_set._cubes

        assert cubes == (cube_, cube_, cube_)

    def it_constructs_its_sequence_of_cube_objects_to_help(self, cube_, _cubedict_):
        cubes_ = (cube_,) * 3
        _cubedict_.return_value = cube_._cube_dict
        transforms_ = [{}, {}, {}]
        cube_set = CubeSet(cubes_, transforms_, None, None)

        cube_sequence = tuple(cube_set._iter_cubes())

        assert len(cube_sequence) == 3
        assert all(isinstance(cube, Cube) for cube in cube_sequence)

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


class DescribeCube(object):
    """Unit-test suite for `cr.cube.cube.Cube` object."""

    def it_provides_the_default_repr_when_enhanced_repr_fails(
        self, dimension_types_prop_
    ):
        dimension_types_prop_.return_value = [1, 2, 3]
        cube = Cube(None, None, None, None)
        cube_repr = cube.__repr__()
        assert cube_repr.startswith("<cr.cube.cube.Cube object at 0x")

    def it_provides_access_to_the_cube_response_dict_to_help(self):
        cube = Cube({"cube": "dict"})
        cube_dict = cube._cube_dict
        assert cube_dict == {"cube": "dict"}

    def and_it_accepts_a_JSON_format_cube_response(self, cube_response_type_fixture):
        cube_response, expected_value = cube_response_type_fixture
        cube = Cube(cube_response)

        cube_dict = cube._cube_dict

        assert cube_dict == expected_value

    def but_it_raises_on_other_cube_response_types(
        self, wrong_cube_response_type_fixtures
    ):
        cube_response, expected_value = wrong_cube_response_type_fixtures
        cube = Cube(cube_response)
        with pytest.raises(TypeError) as e:
            cube._cube_dict

        exception = e.value

        assert str(exception) == expected_value

    def it_knows_if_it_is_mr_by_itself(
        self,
        request,
        cube_dimensions_fixture,
        dimension_types_prop_,
        cube_dimensions_prop_,
    ):
        dimension_types, aliases, expected_value = cube_dimensions_fixture
        cube = Cube(None, None, None, None)
        all_dimensions_ = tuple(
            instance_mock(
                request,
                Dimension,
                name="dim-%d" % idx,
                dimension_type=dt,
                alias=aliases[idx],
            )
            for idx, dt in enumerate(dimension_types)
        )
        apparent_dimensions = _ApparentDimensions(all_dimensions_)
        dimensions = apparent_dimensions._dimensions
        dimension_types_prop_.return_value = dimension_types
        cube_dimensions_prop_.return_value = dimensions

        is_mr_by_itself = cube.is_mr_by_itself

        assert is_mr_by_itself is expected_value

    # fixtures ---------------------------------------------

    @pytest.fixture(
        params=[
            ((DT.CAT, DT.MR), ("alias1", "alias2"), False),
            ((DT.CAT, DT.MR, DT.MR), ("alias1", "alias2", "alias2"), True),
            ((DT.CAT, DT.MR, DT.MR), ("alias1", "alias2", "alias3"), False),
            ((DT.CAT, DT.TEXT, DT.TEXT), ("alias1", "alias2", "alias2"), False),
        ]
    )
    def cube_dimensions_fixture(self, request):
        dimension_types, aliases, expected_value = request.param
        return dimension_types, aliases, expected_value

    @pytest.fixture(
        params=[
            (CR.CAT_X_CAT, CR.CAT_X_CAT.get("value", CR.CAT_X_CAT)),
            ({"value": "val"}, "val"),
        ]
    )
    def cube_response_type_fixture(self, request):
        cube_response, expected_value = request.param
        return cube_response, expected_value

    @pytest.fixture(
        params=[
            (
                None,
                "Unsupported type <NoneType> provided. Cube response must be JSON (str) or dict.",
            ),
            (
                0,
                "Unsupported type <int> provided. Cube response must be JSON (str) or dict.",
            ),
        ]
    )
    def wrong_cube_response_type_fixtures(self, request):
        cube_response, expected_value = request.param
        return cube_response, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def cube_dimensions_prop_(self, request):
        return property_mock(request, Cube, "dimensions")

    @pytest.fixture
    def dimension_types_prop_(self, request):
        return property_mock(request, Cube, "dimension_types")


class DescribeMeasures(object):
    def it_knows_the_population_fraction(self):
        cube_dict, expected_value = (
            {
                "result": {
                    "filtered": {"weighted_n": 10},
                    "unfiltered": {"weighted_n": 9},
                }
            },
            1.1111111111111112,
        )
        measures = _Measures(cube_dict, None)

        population_fraction = measures.population_fraction

        assert population_fraction == expected_value

    def but_the_fraction_is_NaN_for_unfiltered_count_zero(self):
        cube_dict, expected_value = (
            {
                "result": {
                    "filtered": {"weighted_n": 0},
                    "unfiltered": {"weighted_n": 0},
                }
            },
            np.nan,
        )
        measures = _Measures(cube_dict, None)

        population_fraction = measures.population_fraction

        np.testing.assert_equal(population_fraction, expected_value)
