# encoding: utf-8

"""Unit test suite for `cr.cube.cube` module."""

import pytest
import numpy as np

from cr.cube.cube import Cube, CubeSet, _BaseMeasure, _Measures, _MeanMeasure
from cr.cube.cubepart import _Slice, _Strand, _Nub
from cr.cube.dimension import AllDimensions, Dimension
from cr.cube.enums import DIMENSION_TYPE as DT

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---
from ..unitutil import call, class_mock, instance_mock, property_mock


class DescribeCubeSet(object):
    """Unit-test suite for `cr.cube.cube.CubeSet` object."""

    @pytest.mark.parametrize(
        ("cubes_dimtypes", "expected_value"),
        (
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
        ),
    )
    def it_knows_whether_it_can_show_pairwise(
        self, request, cubes_dimtypes, expected_value, _cubes_prop_
    ):
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

    @pytest.mark.parametrize(
        ("first_cube_has_means", "expected_value"), ((True, True), (False, False))
    )
    def it_knows_whether_it_has_means(
        self, first_cube_has_means, expected_value, _cubes_prop_, cube_
    ):
        cube_.has_means = first_cube_has_means
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        has_means = cube_set.has_means

        assert has_means == expected_value

    @pytest.mark.parametrize(
        ("first_cube_has_w_counts", "expected_value"), ((True, True), (False, False))
    )
    def it_knows_whether_it_has_weighted_counts(
        self, first_cube_has_w_counts, expected_value, _cubes_prop_, cube_
    ):
        cube_.is_weighted = first_cube_has_w_counts
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        has_weighted_counts = cube_set.has_weighted_counts

        assert has_weighted_counts == expected_value

    @pytest.mark.parametrize(("ncubes", "expected_value"), ((2, True), (1, False)))
    def it_knows_when_it_is_ca_as_0th(
        self, ncubes, expected_value, _cubes_prop_, cube_
    ):
        cubes_ = (cube_,) * ncubes
        cubes_[0].dimension_types = (DT.CA_SUBVAR,) * ncubes
        _cubes_prop_.return_value = cubes_
        cube_set = CubeSet(cubes_, None, None, None)

        is_ca_as_0th = cube_set.is_ca_as_0th

        assert is_ca_as_0th == expected_value

    @pytest.mark.parametrize(
        ("first_cube_missing_count", "expected_value"), ((34, 34), (0, 0))
    )
    def it_knows_its_missing_count(
        self, first_cube_missing_count, expected_value, _cubes_prop_, cube_
    ):
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

    @pytest.mark.parametrize(
        ("cube_partitions", "expected_value"),
        (
            # --- 3D, 2D, 1D, Nub ---
            ((_Strand, _Slice, _Slice), ((_Strand,), (_Slice,), (_Slice,))),
            ((_Slice, _Slice), ((_Slice,), (_Slice,))),
            ((_Slice,), ((_Slice,),)),
            ((_Nub,), ((_Nub,),)),
        ),
    )
    def it_provides_access_to_the_partition_sets(
        self, cube_partitions, expected_value, _cubes_prop_, cube_
    ):
        cube_.partitions = cube_partitions
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        partition_sets = cube_set.partition_sets

        assert partition_sets == expected_value

    @pytest.mark.parametrize(
        ("population_fraction", "expected_value"),
        ((1.0, 1.0), (0.54, 0.54), (np.nan, np.nan)),
    )
    def it_has_proper_population_fraction(
        self, population_fraction, expected_value, cube_, _cubes_prop_
    ):
        cube_.population_fraction = population_fraction
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        cubeset_population_fraction = cube_set.population_fraction

        np.testing.assert_almost_equal(cubeset_population_fraction, expected_value)

    def it_constructs_its_sequence_of_cube_objects_to_help(
        self, request, Cube_, _is_numeric_mean_prop_
    ):
        cubes_ = tuple(instance_mock(request, Cube) for _ in range(4))
        Cube_.side_effect = iter(cubes_)
        _is_numeric_mean_prop_.return_value = False
        cube_set = CubeSet(
            cube_responses=[{"cube": "resp-1"}, {"cube": "resp-2"}, {"cube": "resp-3"}],
            transforms=[{"xfrms": 1}, {"xfrms": 2}, {"xfrms": 3}],
            population=1000,
            min_base=10,
        )

        cubes = cube_set._cubes

        assert Cube_.call_args_list == [
            call(
                {"cube": "resp-1"},
                cube_idx=0,
                transforms={"xfrms": 1},
                population=1000,
                mask_size=10,
            ),
            call(
                {"cube": "resp-2"},
                cube_idx=1,
                transforms={"xfrms": 2},
                population=1000,
                mask_size=10,
            ),
            call(
                {"cube": "resp-3"},
                cube_idx=2,
                transforms={"xfrms": 3},
                population=1000,
                mask_size=10,
            ),
        ]
        assert cubes == cubes_[:3]

    def but_it_inflates_the_cubes_in_special_case_of_numeric_mean_payload(
        self, request, Cube_, cube_, _is_numeric_mean_prop_
    ):
        cubes_ = tuple(instance_mock(request, Cube) for _ in range(4))
        cube_.inflate.side_effect = iter(cubes_)
        Cube_.return_value = cube_
        _is_numeric_mean_prop_.return_value = True
        cube_set = CubeSet(
            cube_responses=[{"cube": "resp-1"}, {"cube": "resp-2"}, {"cube": "resp-3"}],
            transforms=[{"xfrms": 1}, {"xfrms": 2}, {"xfrms": 3}],
            population=1000,
            min_base=10,
        )

        cubes = cube_set._cubes

        assert Cube_.call_args_list == [
            call(
                {"cube": "resp-1"},
                cube_idx=0,
                transforms={"xfrms": 1},
                population=1000,
                mask_size=10,
            ),
            call(
                {"cube": "resp-2"},
                cube_idx=1,
                transforms={"xfrms": 2},
                population=1000,
                mask_size=10,
            ),
            call(
                {"cube": "resp-3"},
                cube_idx=2,
                transforms={"xfrms": 3},
                population=1000,
                mask_size=10,
            ),
        ]
        assert cube_.inflate.call_args_list == [call(), call(), call()]
        assert cubes == cubes_[:3]

    @pytest.mark.parametrize(
        ("is_multi_cube", "cube_0_ndim", "expected_value"),
        ((False, 1, False), (False, 0, False), (True, 1, False), (True, 0, True)),
    )
    def it_knows_whether_it_is_numeric_mean_to_help(
        self,
        _is_multi_cube_prop_,
        is_multi_cube,
        Cube_,
        cube_,
        cube_0_ndim,
        expected_value,
    ):
        _is_multi_cube_prop_.return_value = is_multi_cube
        cube_.ndim = cube_0_ndim
        Cube_.return_value = cube_
        cube_set = CubeSet(({"cube": 0}, {"cube": 1}), None, None, None)

        is_numeric_mean = cube_set._is_numeric_mean

        assert Cube_.call_args_list == ([call({"cube": 0})] if is_multi_cube else [])
        assert is_numeric_mean == expected_value

    def it_knows_its_valid_counts_summary_to_help(self, _cubes_prop_, cube_):
        cube_.valid_counts_summary = np.array([1, 2, 3])
        cube_.has_means = True
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        valid_counts_summary = cube_set.valid_counts_summary

        np.testing.assert_array_equal(valid_counts_summary, [1, 2, 3])

    def it_knows_its_n_reposnes_to_help(self, _cubes_prop_, cube_):
        cube_.n_responses = 6
        _cubes_prop_.return_value = (cube_,)
        cube_set = CubeSet(None, None, None, None)

        n_responses = cube_set.n_responses

        assert n_responses == 6

    # fixture components ---------------------------------------------

    @pytest.fixture
    def Cube_(self, request):
        return class_mock(request, "cr.cube.cube.Cube")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def _cubes_prop_(self, request):
        return property_mock(request, CubeSet, "_cubes")

    @pytest.fixture
    def _is_multi_cube_prop_(self, request):
        return property_mock(request, CubeSet, "_is_multi_cube")

    @pytest.fixture
    def _is_numeric_mean_prop_(self, request):
        return property_mock(request, CubeSet, "_is_numeric_mean")


class DescribeCube(object):
    """Unit-test suite for `cr.cube.cube.Cube` object."""

    def it_provides_the_default_repr_when_enhanced_repr_fails(
        self, dimension_types_prop_
    ):
        dimension_types_prop_.return_value = [1, 2, 3]
        cube = Cube(None)

        cube_repr = cube.__repr__()

        assert cube_repr.startswith("<cr.cube.cube.Cube object at 0x")

    def it_can_inflate_itself(self, request):
        cube = Cube(
            {"result": {"dimensions": [{"other": "dim"}]}},
            cube_idx=1,
            transforms={"trans": "forms"},
            population=1000,
            mask_size=10,
        )
        # --- mock Cube only *after* constructing cube-under-test; we only want the mock
        # --- to intercept the *second* call
        inflated_cube_ = instance_mock(request, Cube)
        Cube_ = class_mock(request, "cr.cube.cube.Cube", return_value=inflated_cube_)

        inflated_cube = cube.inflate()

        Cube_.assert_called_once_with(
            {
                "result": {
                    "dimensions": [
                        {
                            "references": {"alias": "mean", "name": "mean"},
                            "type": {
                                "class": "categorical",
                                "categories": [{"id": 1, "name": "Mean"}],
                            },
                        },
                        {"other": "dim"},
                    ]
                },
            },
            1,
            {"trans": "forms"},
            1000,
            10,
        )
        assert inflated_cube is inflated_cube_

    @pytest.mark.parametrize(
        ("cube_idx_arg", "expected_value"), ((None, 0), (0, 0), (1, 1), (42, 42))
    )
    def it_knows_its_index_within_its_cube_set(self, cube_idx_arg, expected_value):
        assert Cube(None, cube_idx_arg).cube_index == expected_value

    @pytest.mark.parametrize(
        ("dim_types", "cube_idx", "_is_single_filter_col_cube", "expected_value"),
        (
            ((), 0, False, False),
            ((), 0, True, False),
            ((), 1, True, False),
            ((DT.CA, DT.CAT), 0, False, True),
            ((DT.CA, DT.CAT), 1, True, True),
            ((DT.CAT, DT.CAT), 0, True, False),
            ((DT.CA, DT.CAT, DT.CAT), 0, True, True),
            ((DT.CA, DT.CAT, DT.CAT), 1, False, False),
        ),
    )
    def it_knows_ca_as_0th(
        self,
        request,
        dim_types,
        cube_idx,
        _is_single_filter_col_cube,
        expected_value,
        dimension_types_prop_,
    ):
        property_mock(
            request,
            Cube,
            "_is_single_filter_col_cube",
            return_value=_is_single_filter_col_cube,
        )
        dimension_types_prop_.return_value = dim_types
        cube = Cube(
            response=None,
            cube_idx=cube_idx,
            transforms=None,
            population=None,
            mask_size=0,
        )

        _ca_as_0th = cube._ca_as_0th

        assert _ca_as_0th is expected_value

    @pytest.mark.parametrize(
        ("cube_dict", "expected_value"),
        (({"result": {}}, "Untitled"), ({"result": {"title": "Hipsters"}}, "Hipsters")),
    )
    def it_knows_its_title(self, _cube_dict_prop_, cube_dict, expected_value):
        _cube_dict_prop_.return_value = cube_dict
        assert Cube(None).title == expected_value

    @pytest.mark.parametrize(
        ("cube_dict", "expected_value"),
        (({"result": {}}, False), ({"result": {"is_single_col_cube": True}}, True)),
    )
    def it_knows_if_it_is_a_single_col_filter_cube(
        self, _cube_dict_prop_, cube_dict, expected_value
    ):
        _cube_dict_prop_.return_value = cube_dict
        assert Cube(None)._is_single_filter_col_cube == expected_value

    def it_provides_access_to_the_cube_response_dict_to_help(self):
        assert Cube({"cube": "dict"})._cube_dict == {"cube": "dict"}

    @pytest.mark.parametrize(
        ("cube_response", "expected_value"),
        ((CR.CAT_X_CAT, CR.CAT_X_CAT), ({"value": "val"}, "val")),
    )
    def and_it_accepts_a_JSON_format_cube_response(self, cube_response, expected_value):
        assert Cube(cube_response)._cube_response == expected_value

    @pytest.mark.parametrize(
        ("cube_response", "expected_value"),
        (
            (
                None,
                "Unsupported type <NoneType> provided. Cube response must be JSON "
                "(str) or dict.",
            ),
            (
                0,
                "Unsupported type <int> provided. Cube response must be JSON (str) or "
                "dict.",
            ),
        ),
    )
    def but_it_raises_on_other_cube_response_types(
        self,
        cube_response,
        expected_value,
    ):
        with pytest.raises(TypeError) as e:
            Cube(cube_response)._cube_dict

        assert str(e.value) == expected_value

    @pytest.mark.parametrize(
        "mean_subvariables, mean_references, expected_value",
        (
            (
                ["001", "002"],
                {
                    "subreferences": [
                        {"alias": "A", "name": "A"},
                        {"alias": "B", "name": "B"},
                    ]
                },
                [
                    {"id": 0, "value": {"references": {"alias": "A", "name": "A"}}},
                    {"id": 1, "value": {"references": {"alias": "B", "name": "B"}}},
                ],
            ),
        ),
    )
    def it_knows_its_num_array_dimensions(
        self,
        _mean_references_prop_,
        _mean_subvariables_prop_,
        mean_subvariables,
        mean_references,
        expected_value,
    ):
        _mean_references_prop_.return_value = mean_references
        _mean_subvariables_prop_.return_value = mean_subvariables
        cube = Cube(None)

        _num_array_dimensions = cube._numeric_array_dimension

        assert _num_array_dimensions["type"]["elements"] == expected_value

    def but_it_returns_None_when_mean_subvars_is_empty(self, _mean_subvariables_prop_):
        _mean_subvariables_prop_.return_value = []
        cube = Cube(None)

        _num_array_dimensions = cube._numeric_array_dimension

        assert _num_array_dimensions is None

    @pytest.mark.parametrize(
        "cube_response, expected_value",
        (
            ({}, []),
            ({"foo": "bar"}, []),
            (
                {
                    "result": {
                        "measures": {
                            "mean": {"metadata": {"type": {"subvariables": ["A", "B"]}}}
                        }
                    }
                },
                ["A", "B"],
            ),
        ),
    )
    def it_knows_its_mean_subvariables(
        self, _cube_response_prop_, cube_response, expected_value
    ):
        _cube_response_prop_.return_value = cube_response
        cube = Cube(None)

        mean_subvariables = cube._mean_subvariables

        assert mean_subvariables == expected_value

    @pytest.mark.parametrize(
        "cube_response, expected_value",
        (
            ({}, {}),
            ({"foo": "bar"}, {}),
            (
                {
                    "result": {
                        "measures": {
                            "mean": {
                                "metadata": {
                                    "references": {
                                        "subreferences": [
                                            {"name": "A", "alias": "A"},
                                            {"name": "B", "alias": "B"},
                                        ]
                                    }
                                }
                            }
                        }
                    }
                },
                {
                    "subreferences": [
                        {"name": "A", "alias": "A"},
                        {"name": "B", "alias": "B"},
                    ]
                },
            ),
        ),
    )
    def it_knows_its_mean_references(
        self, _cube_response_prop_, cube_response, expected_value
    ):
        _cube_response_prop_.return_value = cube_response
        cube = Cube(None)

        mean_subreferences = cube._mean_references

        assert mean_subreferences == expected_value

    @pytest.mark.parametrize(
        "cube_response, cube_idx_arg, mean_subvars, num_array_dim, expected_value",
        (
            ({}, None, [], {}, {}),
            ({"result": {"foo": "bar"}}, None, [], {}, {"result": {"foo": "bar"}}),
            (
                {"result": {"foo": "bar"}},
                None,
                ["A", "B"],
                {},
                {"result": {"foo": "bar"}},
            ),
            (
                {"result": {"dimensions": []}},
                None,
                ["A", "B"],
                {"A": "B"},
                {"result": {"dimensions": [{"A": "B"}]}},
            ),
            (
                {"result": {"dimensions": ["A", "B"]}},
                1,
                ["A", "B"],
                {"A": "B"},
                {"result": {"dimensions": [{"A": "B"}, "A", "B"]}},
            ),
        ),
    )
    def it_knows_its_cube_dict(
        self,
        cube_response,
        cube_idx_arg,
        mean_subvars,
        num_array_dim,
        expected_value,
        _cube_response_prop_,
        _mean_subvariables_prop_,
        _numeric_array_dimension_prop_,
    ):
        _cube_response_prop_.return_value = cube_response
        _mean_subvariables_prop_.return_value = mean_subvars
        _numeric_array_dimension_prop_.return_value = num_array_dim
        cube = Cube(None, cube_idx=cube_idx_arg)

        assert cube._cube_dict == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _cube_dict_prop_(self, request):
        return property_mock(request, Cube, "_cube_dict")

    @pytest.fixture
    def _cube_response_prop_(self, request):
        return property_mock(request, Cube, "_cube_response")

    @pytest.fixture
    def _mean_references_prop_(self, request):
        return property_mock(request, Cube, "_mean_references")

    @pytest.fixture
    def _mean_subvariables_prop_(self, request):
        return property_mock(request, Cube, "_mean_subvariables")

    @pytest.fixture
    def _numeric_array_dimension_prop_(self, request):
        return property_mock(request, Cube, "_numeric_array_dimension")

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


class Describe_BaseMeasure(object):
    @pytest.mark.parametrize(
        "cube_idx_arg, dim_types, expected_value",
        (
            (1, (DT.CAT, DT.NUM_ARRAY), True),
            (1, (DT.CAT, DT.CA_SUBVAR), False),
            (1, (DT.NUM_ARRAY,), False),
            (None, (DT.CAT, DT.NUM_ARRAY), False),
        ),
    )
    def it_knows_if_require_array_transposition(
        self, request, cube_idx_arg, dim_types, expected_value
    ):
        all_dimensions = tuple(
            instance_mock(request, Dimension, name="dim-%d" % idx, dimension_type=dt)
            for idx, dt in enumerate(dim_types)
        )

        assert (
            _BaseMeasure({}, all_dimensions, cube_idx_arg).requires_array_transposition
            is expected_value
        )


class Describe_MeanMeasure(object):
    @pytest.mark.parametrize(
        "valid_counts_u, cube_idx_arg, expected_value",
        (
            ([], None, []),
            ([[3, 2, 1], [2, 2, 0]], None, [[3, 2, 1], [2, 2, 0]]),
            ([[3, 2, 1], [2, 2, 0]], 1, [[3, 2, 1], [2, 2, 0]]),
        ),
    )
    def it_knows_its_valid_counts_to_help(
        self,
        request,
        valid_counts_u,
        cube_idx_arg,
        expected_value,
    ):
        _all_dimensions_ = instance_mock(request, AllDimensions)
        _all_dimensions_.shape = (2, 3)
        _mean_measure = _MeanMeasure(
            {
                "result": {
                    "measures": {"valid_count_unweighted": {"data": valid_counts_u}}
                }
            },
            _all_dimensions_,
            cube_idx_arg,
        )

        np.testing.assert_array_equal(_mean_measure.valid_counts, expected_value)
