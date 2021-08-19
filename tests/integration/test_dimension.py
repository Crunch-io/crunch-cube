# encoding: utf-8

"""Integration test suite for the cr.cube.dimension module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.dimension import (
    AllDimensions,
    _AllElements,
    Dimension,
    _Element,
    _ElementTransforms,
    _Subtotal,
)
from cr.cube.enums import DIMENSION_TYPE as DT

from ..fixtures import CR, NA  # ---mnemonic: CR = 'cube-response'---
from ..unitutil import instance_mock, property_mock


class DescribeIntegratedAllDimensions:
    """Integration-test suite for `cr.cube.dimension.AllDimensions` object."""

    @pytest.mark.parametrize(
        "cube_response, expected_types",
        (
            (CR.CAT_X_CAT, (DT.CAT, DT.CAT)),
            (CR.CA_X_MR_WEIGHTED_HS, (DT.CA, DT.CA_CAT, DT.MR, DT.MR_CAT)),
            (
                CR.MR_X_MR_SELECTED_CATEGORIES,
                (DT.MR_SUBVAR, DT.MR_CAT, DT.MR_SUBVAR, DT.MR_CAT),
            ),
        ),
    )
    def it_resolves_the_type_of_each_dimension(self, cube_response, expected_types):
        dimension_dicts = cube_response["result"]["dimensions"]
        all_dimensions = AllDimensions(dimension_dicts)

        dimension_types = tuple(d.dimension_type for d in all_dimensions)

        assert dimension_types == expected_types

    def it_provides_access_to_the_apparent_dimensions(self):
        dimension_dicts = CR.CA_X_MR_WEIGHTED_HS["result"]["dimensions"]
        all_dimensions = AllDimensions(dimension_dicts)

        apparent_dimension_types = tuple(
            d.dimension_type for d in all_dimensions.apparent_dimensions
        )

        assert apparent_dimension_types == (DT.CA, DT.CA_CAT, DT.MR)

    @pytest.mark.parametrize(
        "dim_types, expected_value",
        (
            ((DT.NUM_ARRAY, DT.CAT), (1, 0)),
            ((DT.CAT, DT.CA_SUBVAR), (0, 1)),
            ((DT.NUM_ARRAY, DT.MR_SUBVAR, DT.MR_CAT), (1, 2, 0)),
            ((DT.NUM_ARRAY,), (0,)),
        ),
    )
    def it_knows_if_its_dimensions_order(self, request, dim_types, expected_value):
        _dimensions_ = tuple(
            instance_mock(request, Dimension, name=f"dim-{idx}", dimension_type=dt)
            for idx, dt in enumerate(dim_types)
        )
        _dimensions_prop_ = property_mock(request, AllDimensions, "_dimensions")
        _dimensions_prop_.return_value = _dimensions_

        assert AllDimensions(None).dimension_order == expected_value


class DescribeIntegratedDimension:
    """Integration-test suite for `cr.cube.dimension.Dimension` object."""

    def it_provides_access_to_all_elements_in_its_collection(self, dimension_dict):
        dimension = Dimension(dimension_dict, DT.CAT)

        elements = dimension.all_elements

        assert isinstance(elements, _AllElements)

    def it_knows_its_transformed_description(self, dimension_dict):
        dimension_transforms = {"description": "foobar"}
        dimension = Dimension(dimension_dict, None, dimension_transforms)

        description = dimension.description

        assert description == "foobar"

    def but_it_uses_element_description_when_not_transformed(self, dimension_dict):
        dimension = Dimension(dimension_dict, None)

        description = dimension.description

        assert description == (
            "If President Obama and the Republicans in Congress do not reach a budget"
            " agreement in time to avoid a shutdown of the federal government, who do"
            " you think will more to blame--President Obama or the Republican Congres"
            "s?"
        )

    def it_knows_its_transformed_name(self, dimension_dict):
        dimension_transforms = {"name": "barfoo"}
        dimension = Dimension(dimension_dict, None, dimension_transforms)

        name = dimension.name

        assert name == "barfoo"

    def but_it_uses_the_dimension_name_when_no_transform(self, dimension_dict):
        dimension_transforms = {}
        dimension = Dimension(dimension_dict, None, dimension_transforms)

        name = dimension.name

        assert name == "ShutdownBlame"

    def and_it_uses_alias_when_no_name(self, dimension_dict):
        dimension_dict["references"].pop("name")
        dimension_transforms = {}
        dimension = Dimension(dimension_dict, None, dimension_transforms)

        name = dimension.name

        assert name == "ShutdownBlame"

    def it_knows_whether_it_should_be_pruned(self, dimension_dict):
        dimension_transforms = {"prune": True}
        dimension = Dimension(dimension_dict, None, dimension_transforms)

        prune = dimension.prune

        assert prune is True

    def it_provides_access_to_its_inserted_subtotal_specs(self, dimension_dict):
        dimension_transforms = {}
        dimension = Dimension(dimension_dict, None, dimension_transforms)

        subtotals = dimension.subtotals

        assert len(subtotals) == 1

    def but_it_uses_transforms_insertions_instead_when_present(self, dimension_dict):
        dimension_transforms = {"insertions": []}
        dimension = Dimension(dimension_dict, None, dimension_transforms)

        subtotals = dimension.subtotals

        assert len(subtotals) == 0

    def it_allows_unicode_characters_in_a_subvariable_alias(self):
        slice_ = Cube(CR.CAT_X_MR_UNICODE_SV_ALIAS).partitions[0]
        sv_dimension = slice_._dimensions[1]

        assert sv_dimension.element_ids == ("\u2018dk\u2019", "fi", "is", "no", "se")

    def it_ignores_bad_types_in_transform_dictionary(self):
        transforms = {"columns_dimension": {"elements": {"fi": None, "is": []}}}
        slice_ = Cube(CR.CAT_X_MR, transforms=transforms).partitions[0]
        sv_dimension = slice_._dimensions[1]

        assert sv_dimension.hidden_idxs == tuple()

    # fixture components ---------------------------------------------

    @pytest.fixture
    def dimension_dict(self):
        return CR.ECON_BLAME_WITH_HS["value"]["result"]["dimensions"][0]


class DescribeIntegrated_AllElements:
    """Integration-test suite for `cr.cube.dimension._AllElements` object."""

    def it_constructs_its_element_objects_to_help(self):
        type_dict = Cube(CR.ECON_BLAME_WITH_HS).dimensions[0]._dimension_dict["type"]
        dimension_transforms = {}
        all_elements = _AllElements(type_dict, dimension_transforms, None)

        elements = all_elements._elements

        assert all(isinstance(element, _Element) for element in elements)

    def it_hides_element_objects_by_subvariable_id_to_help(self):
        transforms = {
            "rows_dimension": {
                "elements": {"S2": {"hide": True}},
            },
        }
        all_elements = (
            Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT, transforms=transforms)
            .partitions[0]
            ._dimensions[0]
            .all_elements
        )

        elements = all_elements._elements

        assert len(elements) == 3
        assert elements[1].is_hidden is True


class DescribeIntegrated_Element:
    """Integration-test suite for `cr.cube.dimension._Element` object."""

    def it_knows_its_transformed_label(self, element_dict, element_transforms_):
        element_transforms_.name = "Xfinity Lounge"
        element = _Element(element_dict, None, element_transforms_)

        label = element.label

        assert label == "Xfinity Lounge"

    def but_it_uses_its_base_name_if_no_transform_is_present(
        self, element_dict, element_transforms_
    ):
        element_transforms_.name = None
        element = _Element(element_dict, None, element_transforms_)

        label = element.label

        assert label == "President Obama"

    def it_knows_when_it_is_explicitly_hidden(self, element_dict, element_transforms_):
        element_transforms_.hide = True
        element = _Element(element_dict, None, element_transforms_)

        is_hidden = element.is_hidden

        assert is_hidden is True

    def but_it_is_not_hidden_by_default(self, element_transforms_):
        element_transforms_.hide = None
        element = _Element(None, None, element_transforms_)

        is_hidden = element.is_hidden

        assert is_hidden is False

    # fixture components ---------------------------------------------

    @pytest.fixture
    def element_dict(self):
        return (
            Cube(CR.ECON_BLAME_WITH_HS)
            .dimensions[0]
            ._dimension_dict["type"]["categories"][0]
        )

    @pytest.fixture
    def element_transforms_(self, request):
        return instance_mock(request, _ElementTransforms)


class TestDimension:
    """Legacy integration-test suite for Dimension object."""

    def test_subtotals(self):
        dimension_dict = {
            "references": {
                "view": {
                    "transform": {
                        "insertions": [
                            {"anchor": 101, "name": "This is respondent ideology"},
                            {
                                "anchor": 2,
                                "args": [1, 2],
                                "function": "subtotal",
                                "name": "Liberal net",
                            },
                            {
                                "anchor": 5,
                                "args": [5, 4],
                                "function": "subtotal",
                                "name": "Conservative net",
                            },
                            {
                                "anchor": "fake anchor",
                                "args": ["fake_arg_1", "fake_arg_2"],
                                "function": "fake_fcn_name_not_subtotal",
                                "name": "Fake Name",
                            },
                        ]
                    }
                }
            },
            "type": {
                "categories": [{"id": 1}, {"id": 5}, {"id": 8}, {"id": 9}, {"id": -1}],
                "class": "categorical",
            },
        }
        dimension = Dimension(dimension_dict, DT.CAT)

        subtotals = dimension.subtotals

        assert len(subtotals) == 2

        subtotal = subtotals[0]
        assert isinstance(subtotal, _Subtotal)
        assert subtotal.anchor == "bottom"
        assert subtotal.addend_ids == (1,)
        assert subtotal.label == "Liberal net"

        subtotal = subtotals[1]
        assert isinstance(subtotal, _Subtotal)
        assert subtotal.anchor == 5
        assert subtotal.addend_ids == (5,)
        assert subtotal.label == "Conservative net"

    def test_numeric_values(self):
        dimension_dict = {
            "type": {
                "categories": [
                    {"id": 42, "missing": False, "numeric_value": 1},
                    {"id": 43, "missing": False, "numeric_value": 2},
                    {"id": 44, "missing": True, "numeric_value": 3},
                    {"id": 45, "missing": False, "numeric_value": None},
                    {"id": 46, "missing": False},
                ],
                "class": "categorical",
            }
        }
        dimension = Dimension(dimension_dict, DT.CAT)

        numeric_values = dimension.numeric_values

        assert numeric_values == (1, 2, np.nan, np.nan)
