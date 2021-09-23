# encoding: utf-8

"""Unit test suite for `cr.cube.cubepart` module."""

import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import CubePartition, _Slice, _Strand, _Nub
from cr.cube.dimension import Dimension, _ValidElements
from cr.cube.matrix import Assembler
from cr.cube.stripe.assembler import StripeAssembler

from ..unitutil import class_mock, instance_mock, property_mock


class DescribeCubePartition:
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

    @pytest.mark.parametrize(
        "pw_indices_dict, exception_type, expected_message",
        (
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
        ),
    )
    def but_it_raises_on_invalid_alpha_values(
        self, pw_indices_dict, exception_type, expected_message
    ):
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


class Describe_Slice:
    """Unit test suite for `cr.cube.cubepart._Slice` object."""

    def it_provides_the_default_repr_when_enhanced_repr_fails(self):
        cube = _Slice(None, None, None, None, None)

        cube_repr = cube.__repr__()

        assert cube_repr.startswith("<cr.cube.cubepart._Slice object at 0x")

    @pytest.mark.parametrize(
        "measure_name",
        (
            "column_unweighted_bases",
            "column_weighted_bases",
            "columns_scale_mean_stderr",
            "columns_scale_mean",
            "columns_scale_median",
            "row_proportions",
            "row_unweighted_bases",
            "row_weighted_bases",
            "rows_margin",
            "rows_scale_mean",
            "rows_scale_mean_stddev",
            "rows_scale_mean_stderr",
            "rows_scale_median",
            "table_proportions",
            "table_std_err",
            "table_unweighted_bases",
            "table_weighted_bases",
        ),
    )
    def it_knows_assembled_measures(self, _assembler_prop_, assembler_, measure_name):
        _assembler_prop_.return_value = assembler_
        setattr(assembler_, measure_name, [[1, 2], [3, 4]])

        results = getattr(_Slice(None, None, None, None, None), measure_name)

        assert results == [[1, 2], [3, 4]]

    @pytest.mark.parametrize(
        ("shape", "expected_value"),
        (((4, 2), False), ((4, 0), True), ((0, 2), True), ((0, 0), True)),
    )
    def it_knows_whether_it_is_empty(self, shape, expected_value, shape_prop_):
        shape_prop_.return_value = shape
        slice_ = _Slice(None, None, None, None, None)

        is_empty = slice_.is_empty

        assert is_empty is expected_value

    def but_it_returns_None_when_no_secondary_alpha_specified(
        self, _alpha_alt_prop_, dimension_types_prop_
    ):
        _alpha_alt_prop_.return_value = None
        assert _Slice(None, None, None, None, None).pairwise_indices_alt is None

    def it_knows_the_population_fraction(self, cube_):
        cube_.population_fraction = 0.5
        slice_ = _Slice(cube_, None, None, None, None)

        population_fraction = slice_.population_fraction

        assert population_fraction == 0.5

    def it_provides_the_population_counts(self, _assembler_prop_, assembler_, cube_):
        assembler_.population_proportions = 0.3
        _assembler_prop_.return_value = assembler_
        cube_.population_fraction = 0.1
        population = 20

        slice_ = _Slice(cube_, None, None, population, None)

        assert slice_.population_counts == pytest.approx(0.6)

    def it_provides_the_population_count_moes(
        self, _assembler_prop_, assembler_, cube_
    ):
        assembler_.population_std_err = 0.3
        _assembler_prop_.return_value = assembler_
        cube_.population_fraction = 0.1
        population = 20

        slice_ = _Slice(cube_, None, None, population, None)

        assert slice_.population_counts_moe == pytest.approx(0.6 * 1.959964)

    def it_provides_the_secondary_scale_mean_pairwise_indices(
        self, _alpha_alt_prop_, _only_larger_prop_, PairwiseSignificance_
    ):
        PairwiseSignificance_.scale_mean_pairwise_indices.return_value = (
            (2,),
            (0,),
            (),
        )
        _alpha_alt_prop_.return_value = 0.42
        _only_larger_prop_.return_value = True
        slice_ = _Slice(None, None, None, None, None)

        columns_scale_mean_pw_idxs_alt = slice_.columns_scale_mean_pairwise_indices_alt

        PairwiseSignificance_.scale_mean_pairwise_indices.assert_called_once_with(
            slice_, 0.42, True
        )
        assert columns_scale_mean_pw_idxs_alt == ((2,), (0,), ())

    def but_columns_scale_mean_pairwise_indices_alt_is_None_when_no_secondary_alpha_specified(
        self, _alpha_alt_prop_
    ):
        _alpha_alt_prop_.return_value = None
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.columns_scale_mean_pairwise_indices_alt is None

    @pytest.mark.parametrize(
        "dimensions_dicts, expected_value",
        (
            ([{"references": {}}], ()),
            (
                [
                    {"references": {}},
                    {
                        "references": {
                            "selected_categories": [{"name": "Foo"}, {"name": "Bar"}]
                        }
                    },
                ],
                ("Foo", "Bar"),
            ),
            (
                [
                    {"references": {}},
                    {
                        "references": {
                            "selected_categories": [{"name": "Foo"}, {"id": "Bar"}]
                        }
                    },
                ],
                ("Foo",),
            ),
        ),
    )
    def it_knows_its_selected_category_labels(
        self, _dimensions_prop_, dimensions_dicts, expected_value
    ):
        _dimensions_prop_.return_value = [
            Dimension(dim_dict, None, None) for dim_dict in dimensions_dicts
        ]
        slice_ = _Slice(None, None, None, None, None)

        assert slice_.selected_category_labels == expected_value

    def it_constructs_its_assembler_instance_to_help(
        self, request, cube_, _dimensions_prop_, dimension_, assembler_
    ):
        Assembler_ = class_mock(
            request, "cr.cube.cubepart.Assembler", return_value=assembler_
        )
        _dimensions_prop_.return_value = (dimension_, dimension_)
        slice_idx = 42
        slice_ = _Slice(cube_, slice_idx, None, None, None)

        assembler = slice_._assembler

        Assembler_.assert_called_once_with(cube_, (dimension_, dimension_), slice_idx)
        assert assembler is assembler_

    @pytest.mark.parametrize("ndim, element_ids", ((2, (1, 2, 3)), (3, ())))
    def it_knows_when_its_table_name_is_None(
        self, request, dimension_, _dimensions_prop_, cube_, ndim, element_ids
    ):
        valid_elements_ = instance_mock(request, _ValidElements)
        valid_elements_.element_ids = element_ids
        dimension_.valid_elements = valid_elements_
        cube_.ndim = ndim
        cube_.dimensions = [dimension_]
        slice_ = _Slice(cube_, 0, None, None, None)

        assert slice_.table_name is None

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _alpha_alt_prop_(self, request):
        return property_mock(request, _Slice, "_alpha_alt")

    @pytest.fixture
    def assembler_(self, request):
        return instance_mock(request, Assembler)

    @pytest.fixture
    def _assembler_prop_(self, request):
        return property_mock(request, _Slice, "_assembler")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _dimensions_prop_(self, request):
        return property_mock(request, _Slice, "_dimensions")

    @pytest.fixture
    def dimension_types_prop_(self, request):
        return property_mock(request, _Slice, "dimension_types")

    @pytest.fixture
    def _only_larger_prop_(self, request):
        return property_mock(request, _Slice, "_only_larger")

    @pytest.fixture
    def PairwiseSignificance_(self, request):
        return class_mock(request, "cr.cube.cubepart.PairwiseSignificance")

    @pytest.fixture
    def shape_prop_(self, request):
        return property_mock(request, _Slice, "shape")


class Describe_Strand:
    """Unit test suite for `cr.cube.cubepart._Strand` object."""

    def it_provides_the_default_repr_when_enhanced_repr_fails(self):
        cube = _Strand(None, None, None, None, None, None)

        cube_repr = cube.__repr__()

        assert cube_repr.startswith("<cr.cube.cubepart._Strand object at 0x")

    def it_knows_which_of_its_rows_are_inserted(self, _assembler_prop_, assembler_):
        _assembler_prop_.return_value = assembler_
        assembler_.inserted_row_idxs = (1, 3, 5)
        strand = _Strand(None, None, None, None, None, None)

        assert strand.inserted_row_idxs == (1, 3, 5)

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

    def it_knows_the_population_fraction(self, cube_):
        cube_.population_fraction = 0.5
        strand_ = _Strand(cube_, None, None, None, None, None)

        population_fraction = strand_.population_fraction

        assert population_fraction == 0.5

    def it_provides_the_population_count(
        self,
        cube_,
        _assembler_prop_,
        assembler_,
    ):
        assembler_.population_proportions = 0.3
        _assembler_prop_.return_value = assembler_
        cube_.population_fraction = 0.1
        population = 20
        strand_ = _Strand(cube_, None, population, None, None, None)

        assert strand_.population_counts == pytest.approx(0.6)

    def it_provides_the_population_counts_moe(
        self,
        cube_,
        _assembler_prop_,
        assembler_,
    ):
        assembler_.population_proportion_stderrs = 0.3
        _assembler_prop_.return_value = assembler_
        cube_.population_fraction = 0.1
        population = 20
        strand_ = _Strand(cube_, None, population, None, None, None)

        assert strand_.population_counts_moe == pytest.approx(0.6 * 1.959964)

    def it_knows_its_selected_categories_labels(self, _dimensions_prop_):
        _dimensions_prop_.return_value = [Dimension({"references": {}}, None, None)]
        strand_ = _Strand(None, None, None, None, None, None)

        assert strand_.selected_category_labels == ()

    def it_knows_when_its_table_name_is_None(
        self, request, dimension_, _dimensions_prop_, cube_
    ):
        valid_elements_ = instance_mock(request, _ValidElements)
        valid_elements_.element_ids = ()
        dimension_.valid_elements = valid_elements_
        cube_.dimensions = [dimension_]
        strand = _Strand(cube_, 0, None, None, None, None)

        assert strand.table_name is None

    # fixture components ---------------------------------------------

    @pytest.fixture
    def assembler_(self, request):
        return instance_mock(request, StripeAssembler)

    @pytest.fixture
    def _assembler_prop_(self, request):
        return property_mock(request, _Strand, "_assembler")

    @pytest.fixture
    def cube_(self, request):
        return instance_mock(request, Cube)

    @pytest.fixture
    def dimension_(self, request):
        return instance_mock(request, Dimension)

    @pytest.fixture
    def _dimensions_prop_(self, request):
        return property_mock(request, _Strand, "_dimensions")

    @pytest.fixture
    def shape_prop_(self, request):
        return property_mock(request, _Strand, "shape")


class Describe_Nub:
    """Unit test suite for `cr.cube.cubepart._Nub` object."""

    @pytest.mark.parametrize(
        "unweighted_count, expected_value",
        ((float("NaN"), True), (45.4, False), (0.0, True)),
    )
    def it_knows_when_it_is_empty(self, request, unweighted_count, expected_value):
        property_mock(request, _Nub, "unweighted_count", return_value=unweighted_count)
        nub_ = _Nub(None)

        is_empty = nub_.is_empty

        assert is_empty == expected_value

    def it_knows_its_selected_categories_labels(self):
        nub_ = _Nub(None)

        assert nub_.selected_category_labels == ()
