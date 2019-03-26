# encoding: utf-8

"""Unit test suite for cr.cube.measures.wishart_pairwise_significance module."""

import pytest
import numpy as np

from cr.cube.measures.pairwise_significance import (
    _ColumnPairwiseSignificance,
    PairwiseSignificance,
)
from cr.cube.cube_slice import CubeSlice

from ..unitutil import instance_mock, property_mock


class DescribePairwiseSignificance:
    def it_provides_access_to_its_values(self, request, slice_):
        shape = (2, 2)
        slice_.shape = shape
        expected_test_values = PairwiseSignificance(slice_).values
        assert len(expected_test_values) == 2
        for i, column_pairwise_significance in enumerate(expected_test_values):
            assert column_pairwise_significance._col_idx == i
            assert column_pairwise_significance._slice == slice_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def slice_(self, request):
        return instance_mock(request, CubeSlice)


class Describe_ColumnPairwiseSignificance:
    def it_can_calculate_t_stats(self, t_stats_fixture, slice_):
        col_idx, props, margin, t_stats = t_stats_fixture
        slice_.proportions.return_value = props
        slice_.margin.return_value = margin
        np.testing.assert_almost_equal(
            _ColumnPairwiseSignificance(slice_, col_idx).t_stats, t_stats
        )

    def it_can_calculate_p_vals(self, p_vals_fixture, slice_, t_stats_prop_):
        col_idx, t_stats, margin, p_vals = p_vals_fixture
        slice_.margin.return_value = margin
        t_stats_prop_.return_value = t_stats
        np.testing.assert_almost_equal(
            _ColumnPairwiseSignificance(slice_, col_idx).p_vals, p_vals
        )

    def it_knows_its_pairwise_indices(
        self, pairwise_fixture, p_vals_prop_, t_stats_prop_
    ):
        p_vals, t_stats, only_larger, pairwise_indices = pairwise_fixture
        p_vals_prop_.return_value = p_vals
        t_stats_prop_.return_value = t_stats
        cps = _ColumnPairwiseSignificance(None, None, only_larger=only_larger)
        assert cps.pairwise_indices == pairwise_indices

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (
                [
                    [1, 1.1880841738678649e-02, 3.5437044625837411e-01],
                    [1, 8.6687491806343608e-02, 1.3706553627797602e-02],
                    [1, 3.4774760848677033e-10, 7.8991077081536076e-06],
                    [1, 1.1689409524823668e-02, 4.0907745285876063e-01],
                    [1, 4.6721384259207355e-06, 2.0173609289613204e-03],
                    [1, 6.0764393762673841e-02, 8.4512496679985283e-02],
                    [1, 3.1425637110338300e-06, 1.8450385530011104e-04],
                ],
                [],
                False,
                [(1,), (2,), (1, 2), (1,), (1, 2), (), (1, 2)],
            ),
            (
                [
                    [1, 1.1880841738678649e-02, 3.5437044625837411e-01],
                    [1, 8.6687491806343608e-02, 1.3706553627797602e-02],
                    [1, 3.4774760848677033e-10, 7.8991077081536076e-06],
                    [1, 3.9057646289029502e-01, 2.3909137727474938e-01],
                    [1, 3.1425637110338300e-06, 1.8450385530011104e-04],
                ],
                [
                    [0.0, -2.5161034860255906, -0.9262654193797643],
                    [0.0, -1.7132968783233498, -2.466080736668413],
                    [0.0, -6.281888947702064, -4.474443568845684],
                    [0.0, -0.8586079707543924, -1.1774569464270872],
                    [0.0, 4.663801762560106, 3.743253010905157],
                ],
                True,
                [(1,), (2,), (1, 2), (), ()],
            ),
        ]
    )
    def pairwise_fixture(self, request):
        p_vals, t_stats, only_larger, pairwise_indices = request.param
        return np.array(p_vals), np.array(t_stats), only_larger, pairwise_indices

    @pytest.fixture(
        params=[
            (
                0,
                [
                    [0.0, -2.5161034860255906, -0.9262654193797643],
                    [0.0, -1.7132968783233498, -2.466080736668413],
                    [0.0, -6.281888947702064, -4.474443568845684],
                    [0.0, 2.5218266507892086, 0.8256150660697885],
                    [0.0, 4.581394696943265, 3.089935374237542],
                    [0.0, -1.8754078842945483, -1.7255618024947248],
                    [0.0, -1.1300421773007567, -0.6337152420082218],
                    [0.0, 1.861191038738148, 1.6773069717679185],
                    [0.0, -0.8586079707543924, -1.1774569464270872],
                    [0.0, 4.663801762560106, 3.743253010905157],
                ],
                [2166, 8323, 1419],
                [
                    [1, 1.1880841738678649e-02, 3.5437044625837411e-01],
                    [1, 8.6687491806343608e-02, 1.3706553627797602e-02],
                    [1, 3.4774760848677033e-10, 7.8991077081536076e-06],
                    [1, 1.1689409524823668e-02, 4.0907745285876063e-01],
                    [1, 4.6721384259207355e-06, 2.0173609289613204e-03],
                    [1, 6.0764393762673841e-02, 8.4512496679985283e-02],
                    [1, 2.5848429772787807e-01, 5.2630712639939636e-01],
                    [1, 6.2745164238943607e-02, 9.3569693339442983e-02],
                    [1, 3.9057646289029502e-01, 2.3909137727474938e-01],
                    [1, 3.1425637110338300e-06, 1.8450385530011104e-04],
                ],
            )
        ]
    )
    def p_vals_fixture(self, request):
        col_idx, t_stats, margin, p_vals = request.param
        return col_idx, np.array(t_stats), np.array(margin), np.array(p_vals)

    @pytest.fixture(
        params=[
            (
                0,
                [
                    [0.06832871652816251, 0.053346149225039045, 0.06060606060606061],
                    [0.05124653739612189, 0.04229244262885978, 0.03453136011275546],
                    [0.29778393351800553, 0.22960470984020184, 0.23114869626497533],
                    [0.07617728531855955, 0.09263486723537186, 0.08386187455954898],
                    [0.1768236380424746, 0.21975249309143338, 0.21916842847075405],
                    [0.0443213296398892, 0.035203652529136126, 0.03312191684284708],
                    [0.04524469067405355, 0.03964916496455605, 0.0408738548273432],
                    [0.09187442289935364, 0.10501021266370299, 0.10923185341789993],
                    [0.02723915050784857, 0.02390964796347471, 0.021141649048625793],
                    [0.12096029547553093, 0.1585966598582242, 0.16631430584918958],
                ],
                [2166, 8323, 1419],
                [
                    [0.0, -2.5161034860255906, -0.9262654193797643],
                    [0.0, -1.7132968783233498, -2.466080736668413],
                    [0.0, -6.281888947702064, -4.474443568845684],
                    [0.0, 2.5218266507892086, 0.8256150660697885],
                    [0.0, 4.581394696943265, 3.089935374237542],
                    [0.0, -1.8754078842945483, -1.7255618024947248],
                    [0.0, -1.1300421773007567, -0.6337152420082218],
                    [0.0, 1.861191038738148, 1.6773069717679185],
                    [0.0, -0.8586079707543924, -1.1774569464270872],
                    [0.0, 4.663801762560106, 3.743253010905157],
                ],
            )
        ]
    )
    def t_stats_fixture(self, request):
        col_idx, props, margin, t_stats = request.param
        return col_idx, np.array(props), np.array(margin), np.array(t_stats)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def slice_(self, request):
        return instance_mock(request, CubeSlice)

    @pytest.fixture
    def t_stats_prop_(self, request):
        return property_mock(request, _ColumnPairwiseSignificance, "t_stats")

    @pytest.fixture
    def p_vals_prop_(self, request):
        return property_mock(request, _ColumnPairwiseSignificance, "p_vals")
