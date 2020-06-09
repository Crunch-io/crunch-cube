# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import CubePartition, _Slice, _Strand, _Nub
from cr.cube.dimension import Dimension
from cr.cube.matrix import TransformedMatrix, _VectorAfterHiding
from cr.cube.stripe import _BaseStripeRow, TransformedStripe

from ..unitutil import class_mock, instance_mock, property_mock


class DescribeCubePartition(object):
    """Unit test suite for `cr.cube.cubepart.CubePartition` object."""

    def it_constructs_a_slice_with_its_factory_for_a_2D_cube(self, request, cube_):
        cube_.ndim = 2
        _Slice_ = class_mock(request, "cr.cube.cubepart._Slice")

        slice_ = CubePartition.factory(cube_, 42, {"trans": "forms"}, 1000, False, 10)

        _Slice_.assert_called_once_with(cube_, 42, {"trans": "forms"}, 1000, 10)
        assert slice_ is _Slice_.return_value

    def but_it_constructs_a_strand_for_a_2D_cube_when_ca_as_0th(self, request, cube_):
        cube_.ndim = 2
        _Strand_ = class_mock(request, "cr.cube.cubepart._Strand")

        strand = CubePartition.factory(cube_, 0, ca_as_0th=True)

        _Strand_.assert_called_once_with(cube_, None, None, True, 0, 0)
        assert strand is _Strand_.return_value

    def and_it_constructs_a_strand_for_a_1D_cube(self, request, cube_):
        cube_.ndim = 1
        _Strand_ = class_mock(request, "cr.cube.cubepart._Strand")

        strand = CubePartition.factory(cube_, 42, {"trans": "forms"}, 1000, False, 10)

        _Strand_.assert_called_once_with(cube_, {"trans": "forms"}, 1000, False, 42, 10)
        assert strand is _Strand_.return_value

    def and_it_constructs_a_nub_for_a_0D_cube(self, request, cube_):
        cube_.ndim = 0
        _Nub_ = class_mock(request, "cr.cube.cubepart._Nub")

        nub = CubePartition.factory(cube_)

        _Nub_.assert_called_once_with(cube_)
        assert nub is _Nub_.return_value

    def it_knows_the_index_of_its_cube_in_the_cube_set(self, cube_):
        cube_.cube_index = 42
        cube_partition = CubePartition(cube_)

        cube_index = cube_partition.cube_index

        assert cube_index == 42

    def it_knows_if_cube_is_mr_by_itself(self):
        # --- default of False is overridden by subclasses when appropriate ---
        assert CubePartition(None).cube_is_mr_by_itself is False

    def it_knows_the_primary_alpha_value_to_help(self, _alpha_values_prop_):
        """alpha is the primary confidence-interval threshold specified by the user."""
        _alpha_values_prop_.return_value = (0.042, 0.084)
        assert CubePartition(None)._alpha == 0.042

    @pytest.mark.parametrize(
        "alpha_values, expected_value", (((0.042, 0.084), 0.084), ((0.042, None), None))
    )
    def it_knows_the_secondary_alpha_value_to_help(
        self, _alpha_values_prop_, alpha_values, expected_value
    ):
        _alpha_values_prop_.return_value = alpha_values
        assert CubePartition(None)._alpha_alt == expected_value

    @pytest.mark.parametrize(
        "pw_indices_dict, expected_value",
        (
            # --- default value is .05 ---
            ({}, (0.05, None)),
            ({"alpha": {}}, (0.05, None)),
            ({"alpha": []}, (0.05, None)),
            # --- scalar (float) value sets alpha and sets alpha_alt to None ---
            ({"alpha": 0.025}, (0.025, None)),
            # --- single (float) value sets alpha and sets alpha_alt to None ---
            ({"alpha": [0.03]}, (0.03, None)),
            # --- two values sets alpha to lesser and alpha_alt to greater ---
            ({"alpha": [0.07, 0.03]}, (0.03, 0.07)),
            # --- third and later values are ignored ---
            ({"alpha": (0.07, 0.03, "foobar")}, (0.03, 0.07)),
        ),
    )
    def it_interprets_the_provided_alpha_values_to_help(
        self, pw_indices_dict, expected_value
    ):
        cube_partition = CubePartition(None, {"pairwise_indices": pw_indices_dict})

        alpha_values = cube_partition._alpha_values

        assert alpha_values == expected_value

    def but_it_raises_on_invalid_alpha_values(self, alpha_values_raises_fixture):
        pw_indices_dict, exception_type, expected_message = alpha_values_raises_fixture
        cube_partition = CubePartition(None, {"pairwise_indices": pw_indices_dict})

        with pytest.raises(exception_type) as e:
            cube_partition._alpha_values

        assert str(e.value) == expected_message

    @pytest.mark.parametrize(
        "pw_indices_dict, expected_value",
        (
            # --- default value is True ---
            ({}, True),
            ({"only_larger": "foobar"}, True),
            ({"only_larger": False}, False),
        ),
    )
    def it_knows_the_only_larger_flag_state_to_help(
        self, _transforms_dict_prop_, pw_indices_dict, expected_value
    ):
        _transforms_dict_prop_.return_value = {"pairwise_indices": pw_indices_dict}
        assert CubePartition(None)._only_larger == expected_value

    @pytest.mark.parametrize(
        "transforms, expected_value",
        ((None, {}), ({"trans": "forms"}, {"trans": "forms"})),
    )
    def it_provides_the_transforms_dict_to_help(self, transforms, expected_value):
        """Handles defaulting of transforms arg."""
        assert CubePartition(None, transforms)._transforms_dict == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=(
            # --- type errors ---
            (
                {"alpha": {"al": "pha"}},
                TypeError,
                "transforms.pairwise_indices.alpha, when defined, must be a list of 1 "
                "or 2 float values between 0.0 and 1.0 exclusive. Got %s"
                % repr({"al": "pha"}),
            ),
            (
                {"alpha": "0.05"},
                TypeError,
                "transforms.pairwise_indices.alpha, when defined, must be a list of 1 "
                "or 2 float values between 0.0 and 1.0 exclusive. Got %s"
                % repr("0.05"),
            ),
            # --- scalar out-of-range errors ---
            (
                {"alpha": -0.1},
                ValueError,
                "alpha value, when provided, must be between 0.0 and 1.0 exclusive. "
                "Got %s" % repr(-0.1),
            ),
            (
                {"alpha": 1.0},
                ValueError,
                "alpha value, when provided, must be between 0.0 and 1.0 exclusive. "
                "Got %s" % repr(1.0),
            ),
            # --- sequence value errors ---
            (
                {"alpha": [0.01, ".05"]},
                ValueError,
                "transforms.pairwise_indices.alpha must be a list of 1 or 2 float "
                "values between 0.0 and 1.0 exclusive. Got %s" % repr([0.01, ".05"]),
            ),
        )
    )
    def alpha_values_raises_fixture(self, request):
        pw_indices_dict, exception_type, expected_message = request.param
        return pw_indices_dict, exception_type, expected_message

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _alpha_values_prop_(self, request):
        return property_mock(request, CubePartition, "_alpha_values")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def _transforms_dict_prop_(self, request):
        return property_mock(request, CubePartition, "_transforms_dict")


class Describe_Slice(object):
    """Unit test suite for `cr.cube.cubepart._Slice` object."""

    @pytest.mark.parametrize(
        ("shape", "expected_value"),
        (((4, 2), False), ((4, 0), True), ((0, 2), True), ((0, 0), True)),
    )
    def it_knows_whether_it_is_empty(self, shape, expected_value, shape_prop_):
        shape_prop_.return_value = shape
        slice_ = _Slice(None, None, None, None, None)

        is_empty = slice_.is_empty

        assert is_empty is expected_value

    def it_provides_the_secondary_pairwise_indices(
        self, _alpha_alt_prop_, _only_larger_prop_, PairwiseSignificance_
    ):
        PairwiseSignificance_.pairwise_indices.return_value = [[(0,), (1,)], [(), ()]]
        _alpha_alt_prop_.return_value = 0.42
        _only_larger_prop_.return_value = False
        slice_ = _Slice(None, None, None, None, None)

        pairwise_indices_alt = slice_.pairwise_indices_alt

        PairwiseSignificance_.pairwise_indices.assert_called_once_with(
            slice_, 0.42, False
        )
        assert pairwise_indices_alt == [[(0,), (1,)], [(), ()]]

    def but_it_returns_None_when_no_secondary_alpha_specified(self, _alpha_alt_prop_):
        _alpha_alt_prop_.return_value = None
        assert _Slice(None, None, None, None, None).pairwise_indices_alt is None

    def it_knows_the_population_fraction(self, cube_):
        cube_.population_fraction = 0.5
        slice_ = _Slice(cube_, None, None, None, None)

        population_fraction = slice_.population_fraction

        assert population_fraction == 0.5

    def it_knows_the_row_proportions(self, request, _matrix_prop_, matrix_):
        _matrix_prop_.return_value = matrix_
        matrix_.rows = (
            instance_mock(request, _VectorAfterHiding, proportions=(0.1, 0.2, 0.3)),
            instance_mock(request, _VectorAfterHiding, proportions=(0.4, 0.5, 0.6)),
        )
        slice_ = _Slice(None, None, None, None, None)

        row_proportions = slice_.row_proportions

        np.testing.assert_almost_equal(
            row_proportions, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

    def it_knows_the_rows_margin(self, request, _matrix_prop_, matrix_):
        _matrix_prop_.return_value = matrix_
        matrix_.rows = (
            instance_mock(request, _VectorAfterHiding, margin=(1, 2)),
            instance_mock(request, _VectorAfterHiding, margin=(3, 4)),
        )
        slice_ = _Slice(None, None, None, None, None)

        rows_margin = slice_.rows_margin

        np.testing.assert_almost_equal(rows_margin, [[1, 2], [3, 4]])

    def it_provides_the_secondary_scale_mean_pairwise_indices(
        self, _alpha_alt_prop_, _only_larger_prop_, PairwiseSignificance_
    ):
        PairwiseSignificance_.scale_mean_pairwise_indices.return_value = np.array(
            [(2,), (0,), ()]
        )
        _alpha_alt_prop_.return_value = 0.42
        _only_larger_prop_.return_value = True
        slice_ = _Slice(None, None, None, None, None)

        scale_mean_pairwise_indices_alt = slice_.scale_mean_pairwise_indices_alt

        PairwiseSignificance_.scale_mean_pairwise_indices.assert_called_once_with(
            slice_, 0.42, True
        )
        assert scale_mean_pairwise_indices_alt == ((2,), (0,), ())

    def but_scale_mean_pairwise_indices_alt_is_None_when_no_secondary_alpha_specified(
        self, _alpha_alt_prop_
    ):
        _alpha_alt_prop_.return_value = None
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.scale_mean_pairwise_indices_alt is None

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _alpha_alt_prop_(self, request):
        return property_mock(request, _Slice, "_alpha_alt")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def matrix_(self, request):
        return instance_mock(request, TransformedMatrix)

    @pytest.fixture
    def _matrix_prop_(self, request):
        return property_mock(request, _Slice, "_matrix")

    @pytest.fixture
    def _only_larger_prop_(self, request):
        return property_mock(request, _Slice, "_only_larger")

    @pytest.fixture
    def PairwiseSignificance_(self, request):
        return class_mock(request, "cr.cube.cubepart.PairwiseSignificance")

    @pytest.fixture
    def shape_prop_(self, request):
        return property_mock(request, _Slice, "shape")


class Describe_Strand(object):
    """Unit test suite for `cr.cube.cubepart._Strand` object."""

    def it_knows_which_of_its_rows_are_inserted(self, request, _stripe_prop_, stripe_):
        stripe_.rows = tuple(
            instance_mock(request, _BaseStripeRow, is_inserted=bool(i % 2))
            for i in range(7)
        )
        _stripe_prop_.return_value = stripe_
        strand = _Strand(None, None, None, None, None, None)

        inserted_row_idxs = strand.inserted_row_idxs

        assert inserted_row_idxs == (1, 3, 5)

    @pytest.mark.parametrize(("shape", "expected_value"), (((1,), False), ((0,), True)))
    def it_knows_whether_it_is_empty(self, shape, expected_value, shape_prop_):
        shape_prop_.return_value = shape
        strand = _Strand(None, None, None, None, None, None)

        is_empty = strand.is_empty

        assert is_empty is expected_value

    def it_knows_its_title(self, cube_):
        cube_.title = "Unmarried"
        strand_ = _Strand(cube_, None, None, None, None, None)

        title = strand_.title

        assert title == "Unmarried"

    def it_constructs_its_underlying_stripe_to_help(
        self, request, cube_, _rows_dimension_prop_, dimension_, stripe_
    ):
        TransformedStripe_ = class_mock(request, "cr.cube.cubepart.TransformedStripe")
        TransformedStripe_.stripe.return_value = stripe_
        _rows_dimension_prop_.return_value = dimension_
        strand = _Strand(cube_, None, None, False, 42, None)

        stripe = strand._stripe

        TransformedStripe_.stripe.assert_called_once_with(cube_, dimension_, False, 42)
        assert stripe is stripe_

    def it_knows_the_population_fraction(self, cube_):
        cube_.population_fraction = 0.5
        strand_ = _Strand(cube_, None, None, None, None, None)

        population_fraction = strand_.population_fraction

        assert population_fraction == 0.5

    # fixture components ---------------------------------------------

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _rows_dimension_prop_(self, request):
        return property_mock(request, _Strand, "_rows_dimension")

    @pytest.fixture
    def shape_prop_(self, request):
        return property_mock(request, _Strand, "shape")

    @pytest.fixture
    def stripe_(self, request):
        return instance_mock(request, TransformedStripe)

    @pytest.fixture
    def _stripe_prop_(self, request):
        return property_mock(request, _Strand, "_stripe")


class Describe_Nub(object):
    """Unit test suite for `cr.cube.cubepart._Nub` object."""

    def it_knows_its_data_status(self, _nub_prop_, nub_is_empty_fixture):
        base_count, expected_value = nub_is_empty_fixture
        _nub_prop_.return_value = base_count
        nub_ = _Nub(None)

        is_empty = nub_.is_empty

        assert is_empty == expected_value

    def it_knows_if_cube_is_mr_by_itself(self):
        nub_ = _Nub(None)

        assert nub_.cube_is_mr_by_itself is False

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _nub_prop_(self, request):
        return property_mock(request, _Nub, "base_count")

    # fixtures ---------------------------------------------

    @pytest.fixture(params=[(None, True), (45.4, False)])
    def nub_is_empty_fixture(self, request):
        base_count, expected_value = request.param
        return base_count, expected_value
