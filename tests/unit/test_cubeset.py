# encoding: utf-8

"""Unit test suite for `cr.cube.cube.Cubeset` class."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

from cr.cube.cube import CubeSet
from cr.cube.cubepart import _Slice, _Strand

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---


class TestCrunchCubeSet(object):

    default_transforms = [
        {"columns_dimension": {"insertions": {}}, "rows_dimension": {"insertions": {}}}
    ]

    def it_has_means(self):
        response_sequence = [CR.CAT_X_DATETIME]
        cubeset = CubeSet(
            response_sequence,
            transforms=self.default_transforms,
            population=1000,
            min_base=0,
        )
        mean_measure_dict = (
            CR.CAT_X_DATETIME.get("result", {}).get("measures", {}).get("mean")
        )
        assert cubeset.has_means is False
        assert mean_measure_dict is None

    def it_has_weighted_counts(self):
        response_sequence = [CR.CA_X_MR_WEIGHTED_HS]
        cubeset = CubeSet(
            response_sequence,
            transforms=self.default_transforms,
            population=1000,
            min_base=0,
        )
        unweighted_counts = CR.CA_X_MR_WEIGHTED_HS["result"]["counts"]
        count_data = (
            CR.CA_X_MR_WEIGHTED_HS["result"]["measures"].get("count", {}).get("data")
        )
        assert (unweighted_counts != count_data) is True
        assert cubeset.has_weighted_counts is True

    def it_knows_its_name(self, multiple_response_fixture_with_name_and_description):
        response_sequence = [multiple_response_fixture_with_name_and_description]
        cubeset = CubeSet(
            response_sequence,
            transforms=self.default_transforms,
            population=1000,
            min_base=0,
        )
        assert (
            cubeset.name
            == response_sequence[0]["value"]["result"]["dimensions"][0]["references"][
                "name"
            ]
        )

    def it_can_show_pairwise(self, can_show_pairwise_fixture):
        response_sequence, transforms, expected = can_show_pairwise_fixture
        cubeset = CubeSet(
            response_sequence, transforms=transforms, population=10000, min_base=0
        )
        assert cubeset.can_show_pairwise is expected

    def it_knows_its_missing_count(self):
        response_sequence = [CR.ECON_MEAN_AGE_BLAME_X_GENDER, CR.CAT_X_DATETIME]
        transforms = [self.default_transforms] * 2
        cubeset = CubeSet(
            response_sequence, transforms=transforms, population=10000, min_base=0
        )
        assert cubeset.missing_count == response_sequence[0]["value"]["result"][
            "measures"
        ]["mean"].get("n_missing")

    def it_knows_when_it_is_ca_as_0th(self, is_ca_as_0th_fixture):
        response_sequence, transforms, expected_value = is_ca_as_0th_fixture
        cubeset = CubeSet(
            response_sequence, transforms=transforms, population=10000, min_base=0
        )
        assert cubeset.is_ca_as_0th is expected_value

    def it_has_the_right_partition_set(self, partition_set_fixture):
        response, transforms, expected = partition_set_fixture
        cubeset = CubeSet(response, transforms=transforms, population=10000, min_base=0)
        assert [True] * len(response) == map(
            lambda types: isinstance(types[0], types[1]),
            zip(cubeset.partition_sets[0], expected),
        )

    def it_has_proper_population_fraction(self, population_fraction_fixture):
        """
        Must return 1.0 for an unfiltered cube. Returns `np.nan`
        if the unfiltered count is zero, which would otherwise result in
        a divide - by - zero error.
        """
        response_sequence, expected_value = population_fraction_fixture
        cubeset = CubeSet(
            response_sequence,
            transforms=self.default_transforms,
            population=1000,
            min_base=0,
        )
        np.testing.assert_almost_equal(cubeset.population_fraction, expected_value)

    # ----------------------------------------- fixtures ----------------------------------------------
    @pytest.fixture(params=[CR.CAT_X_DATETIME, CR.CAT_X_CAT_GERMAN_WEIGHTED])
    def multiple_response_fixture_with_name_and_description(self, request):
        return request.param

    def it_knows_its_description(
        self, multiple_response_fixture_with_name_and_description
    ):
        response_sequence = [multiple_response_fixture_with_name_and_description]
        cubeset = CubeSet(
            response_sequence,
            transforms=self.default_transforms,
            population=1000,
            min_base=0,
        )
        assert (
            cubeset.description
            == response_sequence[0]["value"]["result"]["dimensions"][0]["references"][
                "description"
            ]
        )

    @pytest.fixture(
        params=[
            ([CR.CAT_X_CAT_FILT_COMPLETE], 0.576086956522),
            ([CR.CA_SUBVAR_HS_X_MR_X_CA_CAT], 1.0),
            (
                [
                    {
                        "result": {
                            "dimensions": [],
                            "filter_stats": {
                                "filtered_complete": {
                                    "weighted": {
                                        "selected": 0,
                                        "other": 0,
                                        "missing": 1386,
                                    }
                                }
                            },
                        }
                    }
                ],
                np.nan,
            ),
        ]
    )
    def population_fraction_fixture(self, request):
        response_sequence, expected_value = request.param
        return response_sequence, expected_value

    @pytest.fixture(
        # 3D, 2D, 1D
        params=[
            (
                [
                    CR.CA_SUBVAR_HS_X_MR_X_CA_CAT,
                    CR.ECON_MEAN_AGE_BLAME_X_GENDER,
                    CR.CAT_X_DATETIME,
                ],
                [default_transforms] * 3,
                (_Strand, _Slice, _Slice),
            ),
            (
                [CR.CA_SUBVAR_HS_X_MR_X_CA_CAT, CR.ECON_MEAN_AGE_BLAME_X_GENDER],
                default_transforms * 2,
                (_Strand, _Slice),
            ),
            ([CR.CAT_X_DATETIME], default_transforms, (_Slice,)),
        ]
    )
    def partition_set_fixture(self, request):
        response, transforms, expected_value = request.param
        return response, transforms, expected_value

    @pytest.fixture(
        params=[
            (
                [CR.CA_SUBVAR_HS_X_MR_X_CA_CAT, CR.CA_SUBVAR_X_CA_CAT_X_MR],
                [default_transforms] * 2,
                True,
            ),
            ([CR.CA_SUBVAR_HS_X_MR_X_CA_CAT], default_transforms, False),
            (
                [CR.ECON_MEAN_AGE_BLAME_X_GENDER, CR.CAT_X_DATETIME],
                [default_transforms] * 2,
                False,
            ),
        ]
    )
    def is_ca_as_0th_fixture(self, request):
        responses, transforms, expected_value = request.param
        return responses, transforms, expected_value

    @pytest.fixture(
        params=[
            (
                [
                    CR.PAIRWISE_HIROTSU_ILLNESS_X_OCCUPATION,
                    CR.PAIRWISE_HIROTSU_OCCUPATION_X_ILLNESS,
                ],
                [default_transforms] * 2,
                True,
            ),
            ([CR.CAT_X_DATETIME], default_transforms, False),
        ]
    )
    def can_show_pairwise_fixture(self, request):
        responses, transforms, expected_value = request.param
        return responses, transforms, expected_value
