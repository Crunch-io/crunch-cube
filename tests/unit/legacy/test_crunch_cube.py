# encoding: utf-8

"""Unit test suite for the cr.cube.crunch_cube module."""

import numpy as np
import pytest

from unittest import TestCase

from cr.cube.crunch_cube import (
    _BaseMeasure,
    CrunchCube,
    _MeanMeasure,
    _Measures,
    _UnweightedCountMeasure,
    _WeightedCountMeasure,
)
from cr.cube.cube_slice import CubeSlice
from cr.cube.dimension import AllDimensions, _ApparentDimensions, Dimension
from cr.cube.enum import DIMENSION_TYPE as DT

from ...unitutil import (
    class_mock,
    function_mock,
    instance_mock,
    method_mock,
    Mock,
    patch,
    property_mock,
)


class DescribeCrunchCube(object):
    def it_can_compare_pariwise(self, can_compare_pairwise_fixture, slices_prop_):
        slices, can_compare_pairwise = can_compare_pairwise_fixture
        slices_prop_.return_value = slices
        cube = CrunchCube(None)
        assert cube.can_compare_pairwise is can_compare_pairwise

    def it_provides_a_default_repr(self):
        cube = CrunchCube(None)
        repr_ = repr(cube)
        assert repr_.startswith("<cr.cube.crunch_cube.CrunchCube object at 0x")

    def it_knows_its_row_count(self, count_fixture):
        weighted, expected_value = count_fixture
        cube = CrunchCube(None)

        count = cube.count(weighted)

        assert count == expected_value

    def it_provides_access_to_its_dimensions(
        self, _all_dimensions_prop_, all_dimensions_, apparent_dimensions_
    ):
        _all_dimensions_prop_.return_value = all_dimensions_
        all_dimensions_.apparent_dimensions = apparent_dimensions_
        cube = CrunchCube(None)

        dimensions = cube.dimensions

        assert dimensions is apparent_dimensions_

    def it_knows_the_types_of_its_dimension(self, request, dimensions_prop_):
        dimensions_prop_.return_value = tuple(
            instance_mock(request, Dimension, name="dim-%d" % idx, dimension_type=dt)
            for idx, dt in enumerate((DT.CAT, DT.CA_SUBVAR, DT.MR, DT.MR_CAT))
        )
        cube = CrunchCube(None)

        dim_types = cube.dim_types

        assert dim_types == (DT.CAT, DT.CA_SUBVAR, DT.MR, DT.MR_CAT)

    def it_knows_when_it_contains_means_data(
        self, has_means_fixture, _measures_prop_, measures_
    ):
        means, expected_value = has_means_fixture
        _measures_prop_.return_value = measures_
        measures_.means = means
        cube = CrunchCube(None)

        has_means = cube.has_means

        assert has_means is expected_value

    def it_knows_when_it_has_an_mr_dimension(self, has_mr_fixture, mr_dim_ind_prop_):
        mr_dim_indices, expected_value = has_mr_fixture
        mr_dim_ind_prop_.return_value = mr_dim_indices
        cube = CrunchCube(None)

        has_mr = cube.has_mr

        assert has_mr is expected_value

    def it_has_a_deprecated_index_method_that_forwards_to_Index_data(self, request):
        Index_ = class_mock(request, "cr.cube.crunch_cube.Index")
        index_ = Index_.data.return_value
        warn_ = function_mock(request, "cr.cube.crunch_cube.warnings.warn")
        cube = CrunchCube(None)

        index = cube.index()

        warn_.assert_called_once_with(
            "CrunchCube.index() is deprecated. Use CubeSlice.index_table().",
            DeprecationWarning,
        )
        Index_.data.assert_called_once_with(cube, True, False)
        assert index is index_

    def it_knows_when_it_contains_weighted_data(
        self, is_weighted_fixture, _measures_prop_, measures_
    ):
        is_weighted, expected_value = is_weighted_fixture
        _measures_prop_.return_value = measures_
        measures_.is_weighted = is_weighted
        cube = CrunchCube(None)

        is_weighted = cube.is_weighted

        assert is_weighted is expected_value

    def it_knows_its_missing_count(self, _measures_prop_, measures_):
        _measures_prop_.return_value = measures_
        measures_.missing_count = 36
        cube = CrunchCube(None)

        missing = cube.missing

        assert missing == 36

    def it_knows_its_population_fraction(self, _measures_prop_, measures_):
        _measures_prop_.return_value = measures_
        measures_.population_fraction = 0.42
        cube = CrunchCube(None)

        population_fraction = cube.population_fraction

        assert population_fraction == 0.42

    def it_can_adjust_an_axis_to_help(self, request, adjust_fixture, dimensions_prop_):
        dimension_types, axis_cases = adjust_fixture
        dimensions_prop_.return_value = tuple(
            instance_mock(request, Dimension, dimension_type=dimension_type)
            for dimension_type in dimension_types
        )
        cube = CrunchCube(None)

        for axis, expected_value in axis_cases:
            adjusted_axis = cube._adjust_axis(axis)
            assert adjusted_axis == expected_value

    def but_it_raises_on_disallowed_adjustment(self, _is_axis_allowed_):
        _is_axis_allowed_.return_value = False
        axis = 42
        cube = CrunchCube(None)

        with pytest.raises(ValueError):
            cube._adjust_axis(axis)

        _is_axis_allowed_.assert_called_once_with(cube, axis)

    def it_provides_its_AllDimensions_collection_to_help(
        self, AllDimensions_, all_dimensions_
    ):
        cube_response = {"result": {"dimensions": [{"d": 1}, {"d": 2}]}}
        AllDimensions_.return_value = all_dimensions_
        cube = CrunchCube(cube_response)

        all_dimensions = cube._all_dimensions

        AllDimensions_.assert_called_once_with([{"d": 1}, {"d": 2}])
        assert all_dimensions is all_dimensions_

    def it_provides_access_to_the_cube_dict_to_help(self):
        cube = CrunchCube({"cube": "dict"})
        cube_dict = cube._cube_dict
        assert cube_dict == {"cube": "dict"}

    def but_it_first_parses_a_JSON_cube_response(self):
        cube = CrunchCube('{"cubic": "dictum"}')
        cube_dict = cube._cube_dict
        assert cube_dict == {"cubic": "dictum"}

    def and_it_raises_on_invalid_cube_response_type(self):
        cube = CrunchCube(42)
        with pytest.raises(TypeError) as e:
            cube._cube_dict
        assert str(e.value).startswith("Unsupported type")

    def it_knows_whether_an_axis_is_marginable_to_help(
        self, request, allowed_fixture, dimensions_prop_
    ):
        dimension_types, axis_cases = allowed_fixture
        dimensions_prop_.return_value = tuple(
            instance_mock(request, Dimension, dimension_type=dimension_type)
            for dimension_type in dimension_types
        )
        cube = CrunchCube(None)

        for axis, expected_value in axis_cases:
            axis_is_marginable = cube._is_axis_allowed(axis)
            assert axis_is_marginable is expected_value

    def it_selects_the_best_match_counts_measure_to_help(
        self, counts_fixture, _measures_prop_
    ):
        weighted, measures_, expected_measure_ = counts_fixture
        _measures_prop_.return_value = measures_
        cube = CrunchCube(None)

        measure = cube._counts(weighted)

        assert measure is expected_measure_

    def it_selects_the_best_match_measure_to_help(
        self, measure_fixture, _measures_prop_
    ):
        weighted, measures_, expected_measure_ = measure_fixture
        _measures_prop_.return_value = measures_
        cube = CrunchCube(None)

        measure = cube._measure(weighted)

        assert measure is expected_measure_

    def it_provides_access_to_its_Measures_object_to_help(
        self, _all_dimensions_prop_, all_dimensions_, _Measures_, measures_
    ):
        cube_dict = {"cube": "dict"}
        _all_dimensions_prop_.return_value = all_dimensions_
        _Measures_.return_value = measures_
        cube = CrunchCube(cube_dict)

        measures = cube._measures

        _Measures_.assert_called_once_with(cube_dict, all_dimensions_)
        assert measures is measures_

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ([True, True, True], True),
            ([False, True, True], False),
            ([True, True, False], False),
            ([False, False, False], False),
        ]
    )
    def can_compare_pairwise_fixture(self, request):
        slices_can_compare_pairwise, cube_can_compare_pairwise = request.param
        slices = [
            instance_mock(request, CubeSlice, can_compare_pairwise=can_compare_pairwise)
            for can_compare_pairwise in slices_can_compare_pairwise
        ]
        return slices, cube_can_compare_pairwise

    @pytest.fixture(
        params=[
            # ---0 - CAT x CAT---
            (
                (DT.CAT, DT.CAT),
                ((0, (0,)), (1, (1,)), (None, (0, 1)), ((0, 1), (0, 1))),
            ),
            # ---1 - CAT x CAT x CAT---
            (
                (DT.CAT, DT.CAT, DT.CAT),
                ((0, (0,)), (1, (1,)), (2, (2,)), (None, (1, 2)), ((1, 2), (1, 2))),
            ),
            # ---2 - MR (univariate)---
            ((DT.MR,), ((0, (1,)), (None, (1,)))),
            # ---3 - CAT x MR---
            ((DT.CAT, DT.MR), ((0, (0,)), (1, (2,)), (None, (0, 2)), ((0, 1), (0, 2)))),
            # ---4 - MR x CAT---
            ((DT.MR, DT.CAT), ((0, (1,)), (1, (2,)), (None, (1, 2)), ((0, 1), (1, 2)))),
            # ---5 - MR x MR---
            (
                (DT.MR, DT.MR),
                # --col---  --row----  --table-------------------------
                ((0, (1,)), (1, (3,)), (None, (1, 3)), ((0, 1), (1, 3))),
            ),
            # ---6 - CAT x MR x MR---
            (
                (DT.CAT, DT.MR, DT.MR),
                # --0th---  --col----  --row----  --table-------------------------
                ((0, (0,)), (1, (2,)), (2, (4,)), (None, (2, 4)), ((1, 2), (2, 4))),
            ),
            # ---7 - MR x CAT x MR---
            (
                (DT.MR, DT.CAT, DT.MR),
                # --0th---  --col----  --row----  --table-------------------------
                ((0, (1,)), (1, (2,)), (2, (4,)), (None, (2, 4)), ((1, 2), (2, 4))),
            ),
            # ---8 - MR x MR x CAT---
            (
                (DT.MR, DT.MR, DT.CAT),
                # --0th---  --col----  --row----  --table-------------------------
                ((0, (1,)), (1, (3,)), (2, (4,)), (None, (3, 4)), ((1, 2), (3, 4))),
            ),
            # ---9 - CA---
            (
                (DT.CA, DT.CAT),
                # --row-----
                ((1, (1,)),),
            ),
            # ---10 - CA x CAT---
            (
                (DT.CA, DT.CAT, DT.CAT),
                # --row-----
                ((1, (1,)), (2, (2,)), (None, (1, 2)), ((1, 2), (1, 2))),
            ),
            # ---11 - CAT x CA---
            (
                (DT.CAT, DT.CA, DT.CAT),
                # --row-----
                ((2, (2,)),),
            ),
            # ---12 - CA x MR---
            (
                (DT.CA, DT.CAT, DT.MR),
                # --row-----
                ((1, (1,)), (2, (3,)), (None, (1, 3)), ((1, 2), (1, 3))),
            ),
            # ---13 - MR x CAT x CA---
            (
                (DT.MR, DT.CA, DT.CAT),
                # --0th---  --row----
                ((0, (1,)), (2, (3,))),
            ),
        ]
    )
    def adjust_fixture(self, request):
        dimension_types, axis_cases = request.param
        return dimension_types, axis_cases

    @pytest.fixture(
        params=[
            # ---0 - CA---
            ((DT.CA, DT.CAT), ((0, False), (1, True), (None, False))),
            # ---1 - CA x CAT---
            (
                (DT.CA, DT.CAT, DT.CAT),
                ((0, False), (1, True), (2, True), (None, True), ((1, 2), True)),
            ),
            # ---2 - CAT x CA-CAT---
            (
                (DT.CAT, DT.CA, DT.CAT),
                ((0, True), (1, False), (2, True), (None, False), ((1, 2), False)),
            ),
            # ---3 - MR x CA---
            (
                (DT.MR, DT.CA, DT.CAT),
                ((0, True), (1, False), (2, True), (None, False), ((1, 2), False)),
            ),
            # ---4 - CA x MR---
            (
                (DT.CA, DT.CAT, DT.MR),
                ((0, False), (1, True), (2, True), (None, True), ((1, 2), True)),
            ),
            # ---5 - Univariate CAT---
            ((DT.CAT,), ((0, True), (1, True), (None, True))),
            # ---6 - CAT x CAT---
            ((DT.CAT, DT.CAT), ((0, True), (1, True), (None, True), ((0, 1), True))),
            # ---7 - CAT x MR x MR---
            (
                (DT.CAT, DT.MR, DT.MR),
                ((0, True), (1, True), (2, True), (None, True), ((1, 2), True)),
            ),
            # ---8 - MR x CAT x MR---
            (
                (DT.MR, DT.CAT, DT.MR),
                ((0, True), (1, True), (2, True), (None, True), ((1, 2), True)),
            ),
            # ---9 - MR x MR x CAT---
            (
                (DT.MR, DT.MR, DT.CAT),
                ((0, True), (1, True), (2, True), (None, True), ((1, 2), True)),
            ),
        ]
    )
    def allowed_fixture(self, request):
        dimension_types, axis_cases = request.param
        return dimension_types, axis_cases

    @pytest.fixture(params=[(False, 42), (True, 48.732)])
    def count_fixture(self, request, _measures_prop_, measures_):
        weighted, expected_value = request.param
        _measures_prop_.return_value = measures_
        measures_.unweighted_n = 42
        measures_.weighted_n = 48.732
        return weighted, expected_value

    @pytest.fixture(
        params=[
            (False, False, "_UnweightedCountMeasure"),
            (False, True, "_UnweightedCountMeasure"),
            (True, False, "_UnweightedCountMeasure"),
            (True, True, "_WeightedCountMeasure"),
        ]
    )
    def counts_fixture(
        self, request, measures_, weighted_count_measure_, unweighted_count_measure_
    ):
        weighted, is_weighted, expected_type = request.param
        # --weighted indicates the caller has requested that weighted values
        # --be used by passing (weighted=True) to method.
        measures_.weighted_counts = (
            weighted_count_measure_ if is_weighted else unweighted_count_measure_
        )
        measures_.unweighted_counts = unweighted_count_measure_
        expected_measure = {
            "_UnweightedCountMeasure": unweighted_count_measure_,
            "_WeightedCountMeasure": weighted_count_measure_,
        }[expected_type]
        return weighted, measures_, expected_measure

    @pytest.fixture(params=[(True, True), (False, False)])
    def has_means_fixture(self, request, mean_measure_):
        has_mean_measure, expected_value = request.param
        means = mean_measure_ if has_mean_measure else None
        return means, expected_value

    @pytest.fixture(params=[(None, False), (0, True), (3, True), ([2, 5], True)])
    def has_mr_fixture(self, request):
        mr_dim_indices, expected_value = request.param
        return mr_dim_indices, expected_value

    @pytest.fixture(
        params=[
            (True, False, False, "_MeanMeasure"),
            (True, False, True, "_MeanMeasure"),
            (True, True, False, "_MeanMeasure"),
            (True, True, True, "_MeanMeasure"),
            (False, False, False, "_UnweightedCountMeasure"),
            (False, False, True, "_UnweightedCountMeasure"),
            (False, True, False, "_UnweightedCountMeasure"),
            (False, True, True, "_WeightedCountMeasure"),
        ]
    )
    def measure_fixture(
        self,
        request,
        measures_,
        weighted_count_measure_,
        unweighted_count_measure_,
        mean_measure_,
    ):
        has_means, weighted, is_weighted, expected_type = request.param
        # --weighted indicates the caller has requested that weighted values
        # --be used by passing (weighted=True) to method.
        measures_.means = mean_measure_ if has_means else None
        measures_.weighted_counts = (
            weighted_count_measure_ if is_weighted else unweighted_count_measure_
        )
        measures_.unweighted_counts = unweighted_count_measure_
        expected_measure = {
            "_MeanMeasure": mean_measure_,
            "_UnweightedCountMeasure": unweighted_count_measure_,
            "_WeightedCountMeasure": weighted_count_measure_,
        }[expected_type]
        return weighted, measures_, expected_measure

    @pytest.fixture(params=[(True, True), (False, False)])
    def is_weighted_fixture(self, request):
        is_weighted, expected_value = request.param
        return is_weighted, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def AllDimensions_(self, request):
        return class_mock(request, "cr.cube.crunch_cube.AllDimensions")

    @pytest.fixture
    def slices_prop_(self, request):
        return property_mock(request, CrunchCube, "slices")

    @pytest.fixture
    def all_dimensions_(self, request):
        return instance_mock(request, AllDimensions)

    @pytest.fixture
    def _all_dimensions_prop_(self, request):
        return property_mock(request, CrunchCube, "_all_dimensions")

    @pytest.fixture
    def apparent_dimensions_(self, request):
        return instance_mock(request, _ApparentDimensions)

    @pytest.fixture
    def dimensions_prop_(self, request):
        return property_mock(request, CrunchCube, "dimensions")

    @pytest.fixture
    def _is_axis_allowed_(self, request):
        return method_mock(request, CrunchCube, "_is_axis_allowed")

    @pytest.fixture
    def mean_measure_(self, request):
        return instance_mock(request, _MeanMeasure)

    @pytest.fixture
    def _Measures_(self, request):
        return class_mock(request, "cr.cube.crunch_cube._Measures")

    @pytest.fixture
    def measures_(self, request):
        return instance_mock(request, _Measures)

    @pytest.fixture
    def _measures_prop_(self, request):
        return property_mock(request, CrunchCube, "_measures")

    @pytest.fixture
    def mr_dim_ind_prop_(self, request):
        return property_mock(request, CrunchCube, "mr_dim_ind")

    @pytest.fixture
    def unweighted_count_measure_(self, request):
        return instance_mock(request, _UnweightedCountMeasure)

    @pytest.fixture
    def weighted_count_measure_(self, request):
        return instance_mock(request, _WeightedCountMeasure)


class Describe_Measures(object):
    def it_knows_when_a_measure_is_weighted(self, is_weighted_fixture):
        cube_dict, expected_value = is_weighted_fixture
        measures = _Measures(cube_dict, None)

        is_weighted = measures.is_weighted

        assert is_weighted is expected_value

    def it_provides_access_to_the_means_measure(
        self, _MeanMeasure_, mean_measure_, all_dimensions_
    ):
        cube_dict = {"result": {"measures": {"mean": {}}}}
        _MeanMeasure_.return_value = mean_measure_
        measures = _Measures(cube_dict, all_dimensions_)

        means = measures.means

        _MeanMeasure_.assert_called_once_with(cube_dict, all_dimensions_)
        assert means is mean_measure_

    def but_only_when_the_cube_has_a_mean_measure(self):
        measures = _Measures({"result": {"measures": {}}}, None)
        means = measures.means
        assert means is None

    def it_knows_the_missing_count(self, missing_count_fixture, means_prop_):
        means, cube_dict, expected_value = missing_count_fixture
        means_prop_.return_value = means
        measures = _Measures(cube_dict, None)

        missing_count = measures.missing_count

        assert missing_count == expected_value

    def it_knows_the_population_fraction(self, pop_frac_fixture):
        cube_dict, expected_value = pop_frac_fixture
        measures = _Measures(cube_dict, None)

        population_fraction = measures.population_fraction

        # ---works for np.nan, which doesn't equal itself---
        assert population_fraction in (expected_value,)

    def it_provides_access_to_the_unweighted_count_measure(
        self, _UnweightedCountMeasure_, unweighted_count_measure_, all_dimensions_
    ):
        cube_dict = {"cube": "dict"}
        _UnweightedCountMeasure_.return_value = unweighted_count_measure_
        measures = _Measures(cube_dict, all_dimensions_)

        unweighted_counts = measures.unweighted_counts

        _UnweightedCountMeasure_.assert_called_once_with(cube_dict, all_dimensions_)
        assert unweighted_counts is unweighted_count_measure_

    def it_knows_the_unweighted_n(self):
        cube_dict = {"result": {"n": 121}}
        measures = _Measures(cube_dict, None)

        unweighted_n = measures.unweighted_n

        assert unweighted_n == 121

    def it_provides_access_to_the_weighted_count_measure(
        self,
        is_weighted_prop_,
        _WeightedCountMeasure_,
        weighted_count_measure_,
        all_dimensions_,
    ):
        cube_dict = {"cube": "dict"}
        is_weighted_prop_.return_value = True
        _WeightedCountMeasure_.return_value = weighted_count_measure_
        measures = _Measures(cube_dict, all_dimensions_)

        weighted_counts = measures.weighted_counts

        _WeightedCountMeasure_.assert_called_once_with(cube_dict, all_dimensions_)
        assert weighted_counts is weighted_count_measure_

    def but_it_returns_unweighted_count_measure_when_cube_is_not_weighted(
        self, is_weighted_prop_, unweighted_counts_prop_, unweighted_count_measure_
    ):
        is_weighted_prop_.return_value = False
        unweighted_counts_prop_.return_value = unweighted_count_measure_
        measures = _Measures(None, None)

        weighted_counts = measures.weighted_counts

        assert weighted_counts is unweighted_count_measure_

    def it_knows_the_weighted_n(self, weighted_n_fixture):
        cube_dict, is_weighted, expected_value = weighted_n_fixture
        measures = _Measures(cube_dict, None)

        weighted_n = measures.weighted_n

        assert weighted_n == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ({"query": {"weight": "https://x"}}, True),
            ({"weight_var": "weight"}, True),
            ({"weight_url": "https://y"}, True),
            (
                {
                    "result": {
                        "counts": [1, 2, 3],
                        "measures": {"count": {"data": [2, 4, 6]}},
                    }
                },
                True,
            ),
            (
                {
                    "result": {
                        "counts": [1, 2, 3],
                        "measures": {"count": {"data": [1, 2, 3]}},
                    }
                },
                False,
            ),
        ]
    )
    def is_weighted_fixture(self, request):
        cube_dict, expected_value = request.param
        return cube_dict, expected_value

    @pytest.fixture(
        params=[
            ({}, True, 37),
            ({"result": {"missing": 42}}, False, 42),
            ({"result": {}}, False, 0),
        ]
    )
    def missing_count_fixture(self, request, mean_measure_):
        cube_dict, has_means, expected_value = request.param
        mean_measure_.missing_count = expected_value if has_means else None
        means = mean_measure_ if has_means else None
        return means, cube_dict, expected_value

    @pytest.fixture(
        params=[
            ({"result": {}}, 1.0),
            (
                {
                    "result": {
                        "filtered": {"weighted_n": 21},
                        "unfiltered": {"weighted_n": 42},
                    }
                },
                0.5,
            ),
            (
                {
                    "result": {
                        "filtered": {"weighted_n": 0},
                        "unfiltered": {"weighted_n": 0},
                    }
                },
                np.nan,
            ),
            ({"result": {"filtered": {"weighted_n": 43}}}, 1.0),
            ({"result": {"unfiltered": {"weighted_n": 44}}}, 1.0),
        ]
    )
    def pop_frac_fixture(self, request):
        cube_dict, expected_value = request.param
        return cube_dict, expected_value

    @pytest.fixture(
        params=[
            ({}, False, 24.0),
            ({"result": {"measures": {"count": {"data": [7, 9]}}}}, True, 16.0),
        ]
    )
    def weighted_n_fixture(self, request, unweighted_n_prop_, is_weighted_prop_):
        cube_dict, is_weighted, expected_value = request.param
        is_weighted_prop_.return_value = is_weighted
        unweighted_n_prop_.return_value = 24
        return cube_dict, is_weighted, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def all_dimensions_(self, request):
        return instance_mock(request, AllDimensions)

    @pytest.fixture
    def is_weighted_prop_(self, request):
        return property_mock(request, _Measures, "is_weighted")

    @pytest.fixture
    def _MeanMeasure_(self, request):
        return class_mock(request, "cr.cube.crunch_cube._MeanMeasure")

    @pytest.fixture
    def mean_measure_(self, request):
        return instance_mock(request, _MeanMeasure)

    @pytest.fixture
    def means_prop_(self, request):
        return property_mock(request, _Measures, "means")

    @pytest.fixture
    def _UnweightedCountMeasure_(self, request):
        return class_mock(request, "cr.cube.crunch_cube._UnweightedCountMeasure")

    @pytest.fixture
    def unweighted_count_measure_(self, request):
        return instance_mock(request, _UnweightedCountMeasure)

    @pytest.fixture
    def unweighted_counts_prop_(self, request):
        return property_mock(request, _Measures, "unweighted_counts")

    @pytest.fixture
    def unweighted_n_prop_(self, request):
        return property_mock(request, _Measures, "unweighted_n")

    @pytest.fixture
    def _WeightedCountMeasure_(self, request):
        return class_mock(request, "cr.cube.crunch_cube._WeightedCountMeasure")

    @pytest.fixture
    def weighted_count_measure_(self, request):
        return instance_mock(request, _WeightedCountMeasure)


class Describe_BaseMeasure(object):
    def it_provides_access_to_the_raw_cube_array(
        self, _flat_values_prop_, all_dimensions_
    ):
        _flat_values_prop_.return_value = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            2,
            4,
            6,
            8,
            0,
            1,
            3,
            5,
            7,
        )
        all_dimensions_.shape = (2, 3, 3)
        base_measure = _BaseMeasure(None, all_dimensions_)

        raw_cube_array = base_measure.raw_cube_array

        np.testing.assert_array_equal(
            raw_cube_array,
            np.array(
                [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 4, 6], [8, 0, 1], [3, 5, 7]]]
            ),
        )
        assert raw_cube_array.flags.writeable is False

    def it_requires_each_subclass_to_implement_flat_values(self):
        base_measure = _BaseMeasure(None, None)

        with pytest.raises(NotImplementedError) as pt_exc_info:
            base_measure._flat_values

        exception = pt_exc_info.value
        assert str(exception) == "must be implemented by each subclass"

    # fixture components ---------------------------------------------

    @pytest.fixture
    def all_dimensions_(self, request):
        return instance_mock(request, AllDimensions)

    @pytest.fixture
    def _flat_values_prop_(self, request):
        return property_mock(request, _BaseMeasure, "_flat_values")


class Describe_MeanMeasure(object):
    def it_knows_the_missing_count(self, missing_count_fixture):
        cube_dict, expected_value = missing_count_fixture
        mean_measure = _MeanMeasure(cube_dict, None)

        missing_count = mean_measure.missing_count

        assert missing_count == expected_value

    def it_parses_the_flat_values_to_help(self):
        cube_dict = {"result": {"measures": {"mean": {"data": [1, 2, {"?": -1}, 4]}}}}
        mean_measure = _MeanMeasure(cube_dict, None)

        flat_values = mean_measure._flat_values

        assert flat_values == (1, 2, np.nan, 4)

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            ({"result": {"measures": {"mean": {}}}}, 0),
            ({"result": {"measures": {"mean": {"n_missing": 42}}}}, 42),
        ]
    )
    def missing_count_fixture(self, request):
        cube_dict, expected_value = request.param
        return cube_dict, expected_value


class Describe_UnweightedCountMeasure(object):
    def it_parses_the_flat_values_to_help(self):
        cube_dict = {"result": {"counts": [1, 2, 3, 4]}}
        unweighted_count_measure = _UnweightedCountMeasure(cube_dict, None)

        flat_values = unweighted_count_measure._flat_values

        assert flat_values == (1, 2, 3, 4)


class Describe_WeightedCountMeasure(object):
    def it_parses_the_flat_values_to_help(self):
        cube_dict = {"result": {"measures": {"count": {"data": [1.1, 2.2, 3.3, 4.4]}}}}
        weighted_count_measure = _WeightedCountMeasure(cube_dict, None)

        flat_values = weighted_count_measure._flat_values

        assert flat_values == (1.1, 2.2, 3.3, 4.4)


# pylint: disable=invalid-name, no-self-use, protected-access
@patch("cr.cube.crunch_cube.CrunchCube.get_slices", lambda x: None)
class TestCrunchCube(TestCase):
    """Test class for the CrunchCube unit tests.

    This class also tests the functionality of private methods,
    not just the API ones.
    """

    def test_calculate_constraints_sum_axis_0(self):
        prop_table = np.array([[0.32, 0.21, 0.45], [0.12, 0.67, 0.73]])
        prop_cols_margin = np.array([0.2, 0.3, 0.5])
        axis = 0
        expected = np.dot((prop_table * (1 - prop_table)), prop_cols_margin)
        actual = CrunchCube._calculate_constraints_sum(
            prop_table, prop_cols_margin, axis
        )
        np.testing.assert_array_equal(actual, expected)

    def test_calculate_constraints_sum_axis_1(self):
        prop_table = np.array([[0.32, 0.21, 0.45], [0.12, 0.67, 0.73]])
        prop_rows_margin = np.array([0.34, 0.66])
        axis = 1
        expected = np.dot(prop_rows_margin, (prop_table * (1 - prop_table)))
        actual = CrunchCube._calculate_constraints_sum(
            prop_table, prop_rows_margin, axis
        )
        np.testing.assert_array_equal(actual, expected)

    def test_calculate_constraints_sum_raises_value_error_for_bad_axis(self):
        with self.assertRaises(ValueError):
            CrunchCube._calculate_constraints_sum(Mock(), Mock(), 2)

    @patch("cr.cube.crunch_cube.CrunchCube.dimensions", None)
    def test_name_with_no_dimensions(self):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        expected = None
        actual = cube.name
        self.assertEqual(actual, expected)

    @patch("cr.cube.crunch_cube.CrunchCube.dimensions")
    def test_name_with_one_dimension(self, mock_dims):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        mock_dims[0].name = "test"
        expected = "test"
        actual = cube.name
        self.assertEqual(actual, expected)

    @patch("cr.cube.crunch_cube.CrunchCube.dimensions", None)
    def test_description_with_no_dimensions(self):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        expected = None
        actual = cube.name
        self.assertEqual(actual, expected)

    @patch("cr.cube.crunch_cube.CrunchCube.dimensions")
    def test_description_with_one_dimension(self, mock_dims):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        mock_dims[0].description = "test"
        expected = "test"
        actual = cube.description
        self.assertEqual(actual, expected)

    def test_fix_valid_indices_subsequent(self):
        initial_indices = [[1, 2, 3]]
        insertion_index = 2
        expected = [[1, 2, 3, 4]]
        dimension = 0
        actual = CrunchCube._fix_valid_indices(
            initial_indices, insertion_index, dimension
        )
        self.assertEqual(actual, expected)

    def test_fix_valid_indices_with_gap(self):
        initial_indices = [[0, 1, 2, 5, 6]]
        insertion_index = 2
        expected = [[0, 1, 2, 3, 6, 7]]
        dimension = 0
        actual = CrunchCube._fix_valid_indices(
            initial_indices, insertion_index, dimension
        )
        self.assertEqual(actual, expected)

    def test_fix_valid_indices_zero_position(self):
        initial_indices = [[0, 1, 2, 5, 6]]
        insertion_index = -1
        expected = [[0, 1, 2, 3, 6, 7]]
        dimension = 0
        actual = CrunchCube._fix_valid_indices(
            initial_indices, insertion_index, dimension
        )
        self.assertEqual(actual, expected)

    @patch("cr.cube.crunch_cube.CrunchCube.dimensions", [])
    def test_does_not_have_description(self):
        expected = None
        actual = CrunchCube(None).description
        self.assertEqual(actual, expected)

    @patch("cr.cube.crunch_cube.CrunchCube.dimensions")
    def test_has_description(self, mock_dims):
        dims = [Mock(), Mock()]
        mock_dims.__get__ = Mock(return_value=dims)
        expected = dims[0].description
        actual = CrunchCube(None).description
        self.assertEqual(actual, expected)

    def test_test_filter_annotation(self):
        mock_cube = {"filter_names": Mock()}
        expected = mock_cube["filter_names"]
        actual = CrunchCube(mock_cube).filter_annotation
        self.assertEqual(actual, expected)

    def test_margin_pruned_indices_without_insertions(self):
        table = np.array([0, 1, 0, 2, 3, 4])
        expected = np.array([True, False, True, False, False, False])
        actual = CrunchCube._margin_pruned_indices(table, [], 0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_pruned_indices_without_insertions_with_nans(self):
        table = np.array([0, 1, 0, 2, 3, np.nan])
        expected = np.array([True, False, True, False, False, True])
        actual = CrunchCube._margin_pruned_indices(table, [], 0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_pruned_indices_with_insertions(self):
        table = np.array([0, 1, 0, 2, 3, 4])
        insertions = [0, 1]
        expected = np.array([False, False, True, False, False, False])

        actual = CrunchCube._margin_pruned_indices(table, insertions, 0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_pruned_indices_with_insertions_and_nans(self):
        table = np.array([0, 1, 0, 2, 3, np.nan])
        insertions = [0, 1]
        expected = np.array([False, False, True, False, False, True])

        actual = CrunchCube._margin_pruned_indices(table, insertions, 0)
        np.testing.assert_array_equal(actual, expected)

    @patch("numpy.array")
    @patch("cr.cube.crunch_cube.CrunchCube.inserted_hs_indices")
    @patch("cr.cube.crunch_cube.CrunchCube.ndim", 1)
    def test_inserted_inds(self, mock_inserted_hs_indices, mock_np_array):
        expected = Mock()
        mock_np_array.return_value = expected

        cc = CrunchCube(None)

        # Assert indices are not fetched without trasforms
        actual = cc._inserted_dim_inds(None, 0)
        # Assert np.array called with empty list as argument
        mock_np_array.assert_called_once_with([])
        assert actual == expected

        # Assert indices are fetch with transforms
        actual = cc._inserted_dim_inds([1], 0)
        mock_inserted_hs_indices.assert_not_called()
        actual = cc._inserted_dim_inds([0], 0)
        mock_inserted_hs_indices.assert_called_once()
