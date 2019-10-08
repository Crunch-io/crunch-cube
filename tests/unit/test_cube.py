# encoding: utf-8

"""Unit test suite for `cr.cube.cube` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

from cr.cube.cube import Cube, _Measures

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---
from ..unitutil import instance_mock, property_mock


class DescribeCube(object):
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

    # fixtures ---------------------------------------------

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

    @pytest.fixture(
        params=[
            (CR.CAT_X_CAT, CR.CAT_X_CAT.get("value", CR.CAT_X_CAT)),
            ({"value": "val"}, "val"),
        ]
    )
    def cube_response_type_fixture(self, request):
        cube_response, expected_value = request.param
        return cube_response, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

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
