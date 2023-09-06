# encoding: utf-8

"""Unit test suite for cr.cube.dimension module."""

import numpy as np
import pytest

from cr.cube.dimension import (
    Element,
    Elements,
    Dimension,
    Dimensions,
    _ElementIdShim,
    _ElementTransforms,
    _OrderSpec,
    _Subtotal,
    _Subtotals,
)
from cr.cube.enums import (
    COLLATION_METHOD as CM,
    DIMENSION_TYPE as DT,
    MARGINAL,
    MEASURE,
)

from ..unitutil import (
    call,
    class_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class DescribeDimensions:
    """Unit-test suite for `cr.cube.dimension.Dimensions` object."""

    def it_knows_its_shape(self, request):
        dims = Dimensions(
            [
                instance_mock(
                    request, Dimension, name="dim-%d" % idx, shape=element_count
                )
                for idx, element_count in enumerate((3, 2, 1))
            ]
        )
        assert dims.shape == (3, 2, 1)

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (
            ({"type": {"class": "categorical"}}, DT.CAT),
            (
                {"type": {"class": "enum", "subtype": {"class": "variable"}}},
                DT.CA,
            ),
        ),
    )
    def it_parses_the_base_type(self, dimension_dict, expected_value):
        assert Dimensions.dimension_type(dimension_dict) == expected_value

    def but_it_raises_on_unrecognized_type_class(self):
        with pytest.raises(NotImplementedError):
            Dimensions.from_dicts([{"type": {"class": "crunched"}}])

    @pytest.mark.parametrize(
        ("dimension_dict", "expected_value"),
        (
            ({"type": {"class": "categorical", "categories": []}}, DT.CAT),
            (
                {"type": {"class": "categorical", "categories": [{}, {}]}},
                DT.CAT,
            ),
            (
                {"type": {"class": "categorical", "categories": [{"foo": "bar"}, {}]}},
                DT.CAT,
            ),
            (
                {
                    "type": {
                        "class": "categorical",
                        "categories": [{"date": "2019-01"}, {}],
                    }
                },
                DT.CAT_DATE,
            ),
            (
                {
                    "type": {
                        "class": "enum",
                        "subtype": {"class": "variable"},
                        "categories": [{"date": "2019-01"}, {}],
                    }
                },
                DT.CA_SUBVAR,
            ),
        ),
    )
    def and_it_knows_if_it_is_a_categorical_date(self, dimension_dict, expected_value):
        assert Dimensions.dimension_type(dimension_dict) == expected_value

    @pytest.mark.parametrize(
        "typedef, expected_value",
        (
            ({"class": "categorical"}, DT.CAT),
            ({"class": "enum", "subtype": {"class": "variable"}}, DT.CA),
            ({"class": "enum", "subtype": {"class": "datetime"}}, DT.DATETIME),
            ({"class": "enum", "subtype": {"class": "numeric"}}, DT.BINNED_NUMERIC),
            ({"class": "enum", "subtype": {"class": "text"}}, DT.TEXT),
        ),
    )
    def it_determines_the_dimension_type(self, typedef, expected_value):
        assert Dimensions.dimension_type({"type": typedef}) == expected_value

    def but_it_raises_on_unrecognized_base_type(self):
        with pytest.raises(NotImplementedError):
            Dimensions.dimension_type({"type": {"class": "hyper.dimensional"}})

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (
            ({"type": {"class": "categorical"}}, DT.CAT),
            ({"type": {"class": "categorical", "categories": []}}, DT.CAT),
            ({"type": {"class": "categorical", "categories": [{}, {}]}}, DT.CAT),
            (
                {
                    "type": {
                        "class": "categorical",
                        "categories": [{"selected": False}, {}],
                    }
                },
                DT.CAT,
            ),
            (
                {
                    "type": {
                        "class": "categorical",
                        "categories": [
                            {"id": 1, "selected": True},
                            {"id": 0},
                            {"id": -1},
                        ],
                    }
                },
                DT.LOGICAL,
            ),
        ),
    )
    def it_can_tell_when_a_dimension_has_a_selected_category_to_help(
        self, dimension_dict, expected_value
    ):
        assert Dimensions.dimension_type(dimension_dict) == expected_value

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (
            ({"type": {"class": "categorical"}, "references": {}}, DT.CAT),
            (
                {
                    "type": {"class": "categorical"},
                    "references": {"subreferences": {"var": 1}},
                },
                DT.CA_CAT,
            ),
        ),
    )
    def it_distinguishes_an_array_categorical_type_to_help(
        self, dimension_dict, expected_value
    ):
        assert Dimensions.dimension_type(dimension_dict) == expected_value

    @pytest.mark.parametrize(
        "typedef, expected_value",
        (({"class": "enum", "subtype": {"class": "variable"}}, DT.CA),),
    )
    def it_resolves_an_array_type_to_help(self, typedef, expected_value):
        assert Dimensions.dimension_type({"type": typedef}) == expected_value

    @pytest.mark.parametrize(
        "dimdef, expected_value",
        (
            ({"type": {"class": "categorical", "categories": []}}, DT.CAT),
            (
                {"type": {"class": "categorical", "categories": [{"selected": True}]}},
                DT.CAT,
            ),
            (
                {
                    "type": {
                        "class": "categorical",
                        "categories": [
                            {"id": 1, "selected": True},
                            {"id": 0},
                            {"id": -1},
                        ],
                    }
                },
                DT.LOGICAL,
            ),
            (
                {
                    "type": {"class": "categorical", "categories": [{}]},
                    "references": {"subreferences": [{}]},
                },
                DT.CA_CAT,
            ),
            (
                {
                    "type": {
                        "class": "categorical",
                        "categories": [
                            {"id": 1, "selected": True},
                            {"id": 0},
                            {"id": -1},
                        ],
                    },
                    "references": {"subreferences": [{}]},
                },
                DT.MR_CAT,
            ),
            (
                {
                    "type": {
                        "class": "categorical",
                        "categories": [{"selected": True}],
                    },
                    "references": {"subreferences": [{}]},
                },
                DT.CA_CAT,
            ),
        ),
    )
    def it_resolves_a_categorical_type_to_help(self, dimdef, expected_value):
        assert Dimensions.dimension_type(dimdef) == expected_value


class DescribeDimension:
    """Unit-test suite for `cr.cube.dimension.Dimension` object."""

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (
            ({"references": {}}, None),
            ({"references": {"alias": "Elias"}}, "Elias"),
        ),
    )
    def it_knows_its_alias(self, dimension_dict, expected_value):
        assert Dimension(dimension_dict, None).alias == expected_value

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (
            ({"references": {}}, ""),
            ({"references": {"description": None}}, ""),
            ({"references": {"description": ""}}, ""),
            ({"references": {"description": "Crunchiness"}}, "Crunchiness"),
        ),
    )
    def it_knows_its_description(self, dimension_dict, expected_value):
        assert Dimension(dimension_dict, None).description == expected_value

    def it_knows_its_dimension_type(self):
        assert Dimension(None, DT.CAT).dimension_type == DT.CAT

    def it_knows_its_element_ids(self, request, valid_elements_prop_):
        valid_elements_prop_.return_value = (
            instance_mock(request, Element, element_id=(i + 1)) for i in range(6)
        )
        assert Dimension(None, None).element_ids == (1, 2, 3, 4, 5, 6)

    def it_knows_its_element_labels(self, request, valid_elements_prop_):
        valid_elements_prop_.return_value = (
            instance_mock(request, Element, label="lbl %s" % (i + 1)) for i in range(3)
        )
        assert Dimension(None, None).element_labels == ("lbl 1", "lbl 2", "lbl 3")

    def it_computes_its_hidden_idxs(self, request, valid_elements_prop_):
        valid_elements_prop_.return_value = (
            instance_mock(request, Element, is_hidden=bool(i % 2)) for i in range(7)
        )
        assert Dimension(None, None).hidden_idxs == (1, 3, 5)

    def it_knows_its_insertion_ids(self, request):
        property_mock(
            request,
            Dimension,
            "subtotals",
            return_value=tuple(
                instance_mock(request, _Subtotal, insertion_id=(i + 1))
                for i in range(6)
            ),
        )
        assert Dimension(None, None).insertion_ids == (1, 2, 3, 4, 5, 6)

    @pytest.mark.parametrize(
        "dim_refs, dim_xforms, expected_value",
        (
            # --- inheritance is alias <- name <- name-transform. If value is omitted
            # --- (key does not appear in dict), name is inherited. Any present value,
            # --- including None, overrides further inheritance. A `None` value is
            # --- normalized to "" such that return type is always str.
            ({}, {}, ""),
            ({"alias": "A"}, {}, "A"),
            ({"alias": "A", "name": "B"}, {}, "B"),
            ({"alias": "A", "name": None}, {}, ""),
            ({"alias": "A", "name": "B"}, {"name": "C"}, "C"),
            ({"alias": "A", "name": "B"}, {"name": None}, ""),
            ({"name": "B"}, {}, "B"),
            ({"name": None}, {}, ""),
            ({"name": "B"}, {"name": "C"}, "C"),
            ({"name": "B"}, {"name": None}, ""),
            ({}, {"name": "C"}, "C"),
            ({}, {"name": ""}, ""),
            ({}, {"name": None}, ""),
        ),
    )
    def it_knows_its_name(self, dim_refs, dim_xforms, expected_value):
        dimension_dict = {"references": dim_refs}
        dimension = Dimension(dimension_dict, None, dimension_transforms=dim_xforms)
        assert dimension.name == expected_value

    def it_knows_the_numeric_values_of_its_elements(
        self, request, valid_elements_prop_
    ):
        valid_elements_prop_.return_value = tuple(
            instance_mock(request, Element, numeric_value=numeric_value)
            for numeric_value in (1, 2.2, np.nan)
        )
        assert Dimension(None, None).numeric_values == (1, 2.2, np.nan)

    def it_provides_access_to_the_order_spec(self, request):
        order_spec_ = instance_mock(request, _OrderSpec)
        _OrderSpec_ = class_mock(
            request, "cr.cube.dimension._OrderSpec", return_value=order_spec_
        )
        dimension_transforms = {"dim": "xfrms"}
        dimension = Dimension(None, None, dimension_transforms)

        order_spec = dimension.order_spec

        _OrderSpec_.assert_called_once_with(dimension, dimension_transforms)
        assert order_spec is order_spec_

    @pytest.mark.parametrize(
        "dim_transforms, expected_value",
        (
            ({}, False),
            ({"prune": False}, False),
            ({"prune": 1}, False),
            ({"prune": "foobar"}, False),
            ({"prune": True}, True),
        ),
    )
    def it_knows_whether_it_should_be_pruned(self, dim_transforms, expected_value):
        assert Dimension(None, None, dim_transforms).prune is expected_value

    @pytest.mark.parametrize(
        "selected_categories, expected_value",
        (
            ({"selected_categories": None}, ()),
            ({"foo": "bar"}, ()),
            ({}, ()),
            (
                {"selected_categories": [{"name": "Very Fav.", "id": 1}]},
                ({"name": "Very Fav.", "id": 1},),
            ),
        ),
    )
    def it_knows_its_selected_categories(self, selected_categories, expected_value):
        dimension_dict = {"references": selected_categories}
        assert Dimension(dimension_dict, None).selected_categories == expected_value

    @pytest.mark.parametrize(
        "refs, resolution, expected",
        (
            ({"format": {"data": "%Y %b"}}, None, "%Y %b"),
            ({"format": None}, "M", "%b %Y"),
            ({"format": {}}, "Y", "%Y"),
            ({}, "h", "%H:00"),
            ({}, None, None),
        ),
    )
    def it_provides_access_to_element_data_format_to_help(
        self, refs, resolution, expected
    ):
        dimension = Dimension(
            {"references": refs, "type": {"subtype": {"resolution": resolution}}}, None
        )
        assert dimension._element_data_format == expected

    @pytest.mark.parametrize(
        "dimension_dict, insertion_dicts",
        (
            ({}, []),
            ({"references": {}}, []),
            ({"references": {"view": {}}}, []),
            ({"references": {"view": {"transform": {}}}}, []),
            ({"references": {"view": {"transform": {"insertions": []}}}}, []),
            (
                {
                    "references": {
                        "view": {
                            "transform": {
                                "insertions": [
                                    {"insertion": "dict-1", "id": 1},
                                    {"insertion": "dict-2", "id": 2},
                                ]
                            }
                        }
                    }
                },
                [{"insertion": "dict-1", "id": 1}, {"insertion": "dict-2", "id": 2}],
            ),
        ),
    )
    def it_provides_access_to_its_subtotals_to_help(
        self,
        dimension_dict,
        insertion_dicts,
        _Subtotals_,
        subtotals_,
        valid_elements_prop_,
        valid_elements_,
    ):
        dimension_dict["type"] = {"class": "categorical", "categories": []}
        valid_elements_prop_.return_value = valid_elements_
        _Subtotals_.return_value = subtotals_
        dimension = Dimension(dimension_dict, None)

        subtotals = dimension.subtotals

        _Subtotals_.assert_called_once_with(insertion_dicts, valid_elements_)
        assert subtotals is subtotals_

    def it_knows_its_subtotal_labels(self, request, subtotals_prop_):
        subtotals_prop_.return_value = (
            instance_mock(request, _Subtotal, label="lbl %s" % (i + 1))
            for i in range(3)
        )
        assert Dimension(None, None).subtotal_labels == ("lbl 1", "lbl 2", "lbl 3")

    @pytest.mark.parametrize("dimension_type", (DT.MR, DT.CA_SUBVAR))
    def but_it_suppresses_any_subtotals_on_an_array_dimension(
        self,
        request,
        dimension_type,
        _Subtotals_,
        subtotals_,
        valid_elements_prop_,
        valid_elements_,
    ):
        property_mock(request, Dimension, "dimension_type", return_value=dimension_type)
        _Subtotals_.return_value = subtotals_
        valid_elements_prop_.return_value = valid_elements_
        dimension_dict = {"references": {"view": {"transform": {"insertions": [666]}}}}
        dimension = Dimension(dimension_dict, None)

        subtotals = dimension.subtotals

        _Subtotals_.assert_called_once_with([], valid_elements_)
        assert subtotals is subtotals_

    def and_it_overrides_with_subtotal_transforms_when_present(
        self,
        valid_elements_prop_,
        valid_elements_,
        _Subtotals_,
        subtotals_,
    ):
        dimension_transforms = {"insertions": ["subtotal", "dicts"]}
        valid_elements_prop_.return_value = valid_elements_
        _Subtotals_.return_value = subtotals_
        dimension = Dimension(None, None, dimension_transforms)

        subtotals = dimension.subtotals

        _Subtotals_.assert_called_once_with(
            ["subtotal", "dicts"], valid_elements_, False
        )
        assert subtotals is subtotals_

    @pytest.mark.parametrize(
        "dim_type, view_insertion_dicts, dimension_transforms_dict, insertion_dicts",
        (
            (DT.MR, [], {}, []),
            (DT.CA_SUBVAR, [], {}, []),
            (
                DT.CAT,
                [{"anchor": "top", "name": "a", "id": 1}],
                {"insertions": [{"anchor": "3", "name": "a", "hide": True, "id": 1}]},
                [{"anchor": "top", "name": "a", "id": 1}],
            ),
            (
                DT.CAT,
                [
                    {"anchor": "top", "name": "a", "id": 1},
                    {"anchor": "bottom", "name": "b", "hide": True, "id": 2},
                ],
                {
                    "insertions": [
                        {"anchor": "3", "name": "a", "hide": True, "id": 1},
                        {"anchor": "2", "name": "b", "hide": True, "id": 2},
                    ]
                },
                [
                    {"anchor": "top", "name": "a", "id": 1},
                    {"anchor": "bottom", "name": "b", "id": 2, "hide": True},
                ],
            ),
            (
                DT.CAT,
                [{"anchor": "top", "name": "a", "id": 1}],
                {},
                [{"anchor": "top", "name": "a", "id": 1}],
            ),
            (
                DT.CAT,
                [
                    {"anchor": 1, "name": "a", "id": 1},
                    {"anchor": 1, "name": "b", "id": 2},
                ],
                {
                    "insertions": [
                        {"anchor": 1, "name": "b", "id": 1},
                        {"anchor": 1, "name": "a", "id": 2},
                    ]
                },
                [
                    {"anchor": 1, "name": "a", "id": 1},
                    {"anchor": 1, "name": "b", "id": 2},
                ],
            ),
        ),
    )
    def it_provides_access_to_its_subtotals_in_payload_order_to_help(
        self,
        request,
        dim_type,
        view_insertion_dicts,
        dimension_transforms_dict,
        insertion_dicts,
        _Subtotals_,
        subtotals_,
        valid_elements_prop_,
        valid_elements_,
    ):
        valid_elements_prop_.return_value = valid_elements_
        _Subtotals_.return_value = subtotals_
        dimension = Dimension(None, None)
        property_mock(
            request,
            Dimension,
            "_view_insertion_dicts",
            return_value=view_insertion_dicts,
        )
        property_mock(
            request,
            Dimension,
            "_dimension_transforms_dict",
            return_value=dimension_transforms_dict,
        )
        property_mock(request, Dimension, "dimension_type", return_value=dim_type)

        subtotals = dimension.subtotals_in_payload_order

        _Subtotals_.assert_called_once_with(insertion_dicts, valid_elements_)
        assert subtotals is subtotals_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _Subtotals_(self, request):
        return class_mock(request, "cr.cube.dimension._Subtotals")

    @pytest.fixture
    def subtotals_(self, request):
        return instance_mock(request, _Subtotals)

    @pytest.fixture
    def subtotals_prop_(self, request):
        return property_mock(request, Dimension, "subtotals")

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, Elements)

    @pytest.fixture
    def valid_elements_prop_(self, request):
        return property_mock(request, Dimension, "valid_elements")


class DescribeElements:
    """Unit-test suite for `cr.cube.dimension.Elements` object."""

    def it_knows_the_element_ids(self, request):
        elements = Elements(
            instance_mock(request, Element, element_id=n) for n in (1, 2, 5)
        )
        assert elements.element_ids == (1, 2, 5)

    def it_knows_the_element_indices(self, request):
        elements = Elements(
            instance_mock(request, Element, index=index) for index in (1, 3, 4)
        )
        assert elements.element_idxs == (1, 3, 4)

    def it_can_find_an_element_by_id(self, request, _elements_by_id_prop_):
        elements_ = tuple(
            instance_mock(request, Element, element_id=element_id)
            for element_id in (3, 7, 11)
        )
        _elements_by_id_prop_.return_value = {
            element_.element_id: element_ for element_ in elements_
        }
        elements = Elements()

        element = elements.get_by_id(7)

        assert element is elements_[1]

    def it_maintains_a_dict_of_elements_by_id_to_help(self, request):
        elements = Elements(
            instance_mock(request, Element, element_id=element_id)
            for element_id in (4, 6, 7)
        )
        assert elements._elements_by_id == {
            4: elements[0],
            6: elements[1],
            7: elements[2],
        }

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _elements_by_id_prop_(self, request):
        return property_mock(request, Elements, "_elements_by_id")

    def it_provides_access_to_valid_elements(self, request):
        all_elements = Elements(
            instance_mock(request, Element, name="el%s" % idx, missing=idx >= 3)
            for idx in range(5)
        )
        assert all_elements.valid_elements == all_elements[:3]

    @pytest.mark.parametrize(
        "dim_xforms, element_dicts, dim_type, expected_value",
        (
            (
                {"insertions": [{"name": "A&B", "hide": True}]},
                [
                    {"id": 1, "value": {"id": "A&B"}},
                    {"id": 2, "value": {"id": "0004"}},
                ],
                DT.MR,
                {1: {"hide": True}},
            ),
            (
                {
                    "insertions": [{"name": "A&B", "hide": True}],
                    "elements": {2: {"hide": True}},
                },
                [
                    {"id": 1, "value": {"id": "A&B"}},
                    {"id": 2, "value": {"id": "0004"}},
                ],
                DT.MR,
                {1: {"hide": True}, 2: {"hide": True}},
            ),
            (
                {
                    "insertions": [{"name": "A&B", "hide": True}],
                    "elements": {2: {"hide": True}},
                },
                [
                    {"id": 1, "value": {"id": "A&B"}},
                    {"id": 2, "value": {"id": "0004"}},
                ],
                DT.CAT,
                {2: {"hide": True}},  # --- CAT insertions don't get updated
            ),
        ),
    )
    def it_knows_its_mr_insertions_elements_transforms_from_transforms_dict(
        self, dim_xforms, element_dicts, dim_type, expected_value
    ):
        elements = Elements.from_typedef(
            {"class": "categorical", "categories": element_dicts},
            dim_xforms,
            dim_type,
            None,
        )
        assert {
            e.element_id: {"hide": True} for e in elements if e.is_hidden
        } == expected_value

    @pytest.mark.parametrize(
        "resolution, out_format, x, expected",
        (
            ("Y", "%Y-%m-%d", "2023", "2023-01-01"),
            ("M", "%m/%d/%Y", "2022-08", "08/01/2022"),
            ("D", "%d/%m/%Y", "1999-12-31", "31/12/1999"),
            ("h", "%d %B, %Y", "2023-06-06T11", "06 June, 2023"),
            ("m", "%d %B, %Y", "2023-07-04T09:20", "04 July, 2023"),
            ("s", "%d %b %Y", "2023-12-01T09:20:04", "01 Dec 2023"),
            ("ms", "%B %Y", "2022-06-22T09:48:35.921", "June 2022"),
            ("ms", "%Y", "2020-03-22T09:15:03.237821", "2020"),
            ("D", "%b %Y", "2021-08-11", "Aug 2021"),
            ("M", "%d %B %Y", "2023-11", "01 November 2023"),
            # --- Default formats based on resolution
            ("Y", "%Y", "2023", "2023"),
            ("M", "%b %Y", "2022-08", "Aug 2022"),
            ("D", "%Y W%W", "1999-12-31", "1999 W52"),
            ("D", "%d %b %Y", "1999-12-31", "31 Dec 1999"),
            ("h", "%H:00", "2023-06-06T11", "11:00"),
            ("m", "%H:%M", "2023-07-04T09:20", "09:20"),
            ("s", ":%S", "2023-12-01T09:20:04", ":04"),
            # --- Edge cases
            (None, "%Y", "2023", "2023"),  # ------------ No in_format, no change
            ("Y", None, "2023", "2023"),  # ----------- No out_format, no change
            ("Y", "%Y-%m-%d", "2023-01", "2023-01"),  # --- invalid x, no change
            ("M", "%Y-%m-%d", "2023", "2023"),  # ------- invalid x, no change
            ("Y", "%Y", "abc", "abc"),  # ----------------- invalid x, no change
        ),
    )
    def it_can_format_datetime_label_to_help(
        self, request, resolution, out_format, x, expected
    ):
        all_elements = Elements.from_typedef(
            {
                "class": "enum",
                "subtype": {"class": "datetime", "resolution": resolution},
                "elements": [{"id": 1, "name": "foo"}],
            },
            {},
            DT.DATETIME,
            out_format,
        )
        assert all_elements[0]._label_formatter(x) == expected

    @pytest.mark.parametrize(
        "dim_type, value, expected",
        ((DT.CATEGORICAL, "cat a", "cat a"), (DT.BINNED_NUMERIC, 123, "123")),
    )
    def and_it_can_format_non_datetime_element_label(self, dim_type, value, expected):
        typedef = {
            "class": "enum",
            "subtype": {"class": "numeric"},
            "elements": [{"id": 1, "name": "foo"}],
        }
        all_elements = Elements.from_typedef(typedef, {}, dim_type, None)
        assert all_elements[0]._label_formatter(value) == expected

    def it_obeys_category_order_from_typedef(self):
        # Unordered
        all_elements = Elements.from_typedef(
            {
                "class": "categorical",
                "categories": [{"id": 1}, {"id": 2}, {"id": 3}],
            },
            {},
            DT.CAT,
            None,
        )
        assert all_elements.element_ids == (1, 2, 3)

        # Ordered
        all_elements = Elements.from_typedef(
            {
                "class": "categorical",
                "categories": [{"id": 1}, {"id": 2}, {"id": 3}],
                "order": [2, 3, 1],
            },
            {},
            DT.CAT,
            None,
        )
        assert all_elements.element_ids == (2, 3, 1)


class Describe_ValidElements:
    """Unit-test suite for `cr.cube.dimension.Elements.valid_elements`."""

    def it_gets_its_Element_objects_from_anElements_object(self, request):
        all_elements = Elements(
            instance_mock(request, Element, name="element-%s" % idx, missing=missing)
            for idx, missing in enumerate([False, True, False])
        )
        assert all_elements.valid_elements == (all_elements[0], all_elements[2])


class Describe_ElementIdShim:
    """Unit-test suite for `cr.cube.dimension._ElementIdShim` object."""

    def it_provides_the_shimmed_dimension_dict_for_subvars(self, _subvar_aliases_prop_):
        _subvar_aliases_prop_.return_value = tuple(("alias1", "alias2"))
        dimension_dict = {
            "type": {
                "elements": [{"id": 1, "element_info": 100}, {"id": 2}],
                "other": "in type",
            },
            "another": "outside type",
        }
        shim_ = _ElementIdShim(DT.MR_SUBVAR, dimension_dict, None)

        assert shim_.shimmed_dimension_dict == {
            "type": {
                "elements": [{"id": "alias1", "element_info": 100}, {"id": "alias2"}],
                "other": "in type",
            },
            "another": "outside type",
        }

    def and_it_provides_the_shimmed_dimension_dict_for_datetime(self):
        dimension_dict = {
            "type": {
                "elements": [
                    {"id": 1, "value": "val1", "element_info": 100},
                    {"id": 2, "value": {"?": -1}},
                    {"id": 3, "value": "another val"},
                ],
                "other": "in type",
            },
            "another": "outside type",
        }
        shim_ = _ElementIdShim(DT.DATETIME, dimension_dict, None)

        assert shim_.shimmed_dimension_dict == {
            "type": {
                "elements": [
                    {"id": "val1", "value": "val1", "element_info": 100},
                    {"id": 2, "value": {"?": -1}},
                    {"id": "another val", "value": "another val"},
                ],
                "other": "in type",
            },
            "another": "outside type",
        }

    def but_other_types_of_dimensions_are_left_alone(self):
        dimension_dict = {"dimension": "dictionary"}
        shim_ = _ElementIdShim(DT.CAT, dimension_dict, None)

        assert shim_.shimmed_dimension_dict == dimension_dict

    def it_provides_the_shimmed_dimension_transform_dict_for_subvars(
        self, _replaced_element_transforms_, _replaced_order_element_ids_
    ):
        dimension_transforms_dict = {
            "elements": "all_elements",
            "order": {"element_ids": "a", "ignored": "string"},
            "another": "ignored",
        }
        _replaced_element_transforms_.return_value = {"replaced": "element_transforms"}
        _replaced_order_element_ids_.return_value = {"replaced": "ids"}
        shim_ = _ElementIdShim(DT.MR_SUBVAR, None, dimension_transforms_dict)

        assert shim_.shimmed_dimension_transforms_dict == {
            "elements": {"replaced": "element_transforms"},
            "order": {"element_ids": {"replaced": "ids"}, "ignored": "string"},
            "another": "ignored",
        }
        _replaced_element_transforms_.assert_called_once_with(shim_, "all_elements")
        _replaced_order_element_ids_.assert_called_once_with(shim_, "a")

    def but_it_skips_replacing_if_not_present(
        self, _replaced_element_transforms_, _replaced_order_element_ids_
    ):
        dimension_transforms_dict = {
            "order": {"ignored": "string"},
            "another": "ignored",
        }
        shim_ = _ElementIdShim(DT.MR_SUBVAR, None, dimension_transforms_dict)

        assert shim_.shimmed_dimension_transforms_dict == dimension_transforms_dict
        _replaced_element_transforms_.assert_not_called()
        _replaced_order_element_ids_.assert_not_called()

    def and_it_skips_replacing_if_not_subvars(
        self, _replaced_element_transforms_, _replaced_order_element_ids_
    ):
        dimension_transforms_dict = {
            "elements": {"replaced": "element_transforms"},
            "order": {"element_ids": {"replaced": "ids"}, "ignored": "string"},
            "another": "ignored",
        }
        shim_ = _ElementIdShim(DT.CAT, None, dimension_transforms_dict)

        assert shim_.shimmed_dimension_transforms_dict == dimension_transforms_dict
        _replaced_element_transforms_.assert_not_called()
        _replaced_order_element_ids_.assert_not_called()

    @pytest.mark.parametrize(
        "id, subvar_aliases, subvar_ids, raw_element_ids, expected",
        (
            ("all", ("all", "al1", "al2"), ("01", "all", "03"), ("1", 3, "all"), "all"),
            ("03", ("all", "al1", "al2"), ("01", "all", "03"), ("1", 3, "all"), "al2"),
            ("1", ("all", "al1", "al2"), ("01", "all", "03"), ("1", 3, "all"), "all"),
            (3, ("all", "al1", "al2"), ("01", "all", "03"), ("1", 3, "all"), "al1"),
            (2, ("all", "al", "al2"), ("01", "all", "03"), ("1", 3, "all"), "al2"),
            (21, ("all", "al1", "al2"), ("01", "all", "03"), ("1", 3, "all"), None),
            ("xyz", ("all", "al1", "al2"), ("01", "all", "03"), ("1", 3, "all"), None),
            ("1", ("all", "al1", "al2"), ("001", "1", "003"), ("1", 3, "all"), "all"),
            ("1", ("all", "al1", "al2"), ("001", "1", "003"), (1, 3, "all"), "al1"),
        ),
    )
    def it_translates_element_ids_for_subvariables(
        self,
        _raw_element_id_prop_,
        _subvar_aliases_prop_,
        _subvar_ids_prop_,
        id,
        subvar_aliases,
        subvar_ids,
        raw_element_ids,
        expected,
    ):
        _subvar_aliases_prop_.return_value = subvar_aliases
        _subvar_ids_prop_.return_value = subvar_ids
        _raw_element_id_prop_.return_value = raw_element_ids

        shim_ = _ElementIdShim(DT.MR_SUBVAR, {}, None)

        assert shim_.translate_element_id(id) == expected

    def but_it_leaves_non_array_ids_alone(self):
        _ElementIdShim(DT.CAT, None, None).translate_element_id(25) == 25

    @pytest.mark.parametrize(
        "id_, expected",
        (
            (1, "2023"),
            ("0", "2022"),
            ("a", "a"),
            ("2021", "2021"),
        ),
    )
    def and_it_translates_for_datetime(self, request, id_, expected):
        property_mock(
            request,
            _ElementIdShim,
            "_element_values_dict",
            return_value={0: "2022", 1: "2023"},
        )
        shim = _ElementIdShim(DT.DATETIME, None, None)
        assert shim.translate_element_id(id_) == expected

    def it_provides_the_raw_element_ids_to_help(self):
        dimension_dict = {"type": {"elements": [{"id": 1}, {"id": "b"}]}}
        shim_ = _ElementIdShim(None, dimension_dict, None)

        assert shim_._raw_element_ids == tuple((1, "b"))

    def it_replaces_the_element_transforms_to_help(
        self, _raw_element_id_prop_, _subvar_aliases_prop_, _subvar_ids_prop_
    ):
        element_transforms = {
            "xxx": {"transform": "object"},
            "abc": {"another": "transform"},
        }
        _raw_element_id_prop_.return_value = tuple((1,))
        _subvar_aliases_prop_.return_value = tuple(("alias2",))
        _subvar_ids_prop_.return_value = tuple(("abc",))

        shim_ = _ElementIdShim(DT.MR_SUBVAR, {}, None)

        assert shim_._replaced_element_transforms(element_transforms) == {
            "alias2": {"another": "transform"}
        }

    def it_replaces_the_element_transforms_to_help_with_alias_keys(self):
        element_transforms = {
            "alias1": {"transform": "object"},
            "alias2": {"another": "transform"},
            "key": "alias",
        }
        shim_ = _ElementIdShim(None, None, None)

        assert (
            shim_._replaced_element_transforms(element_transforms) == element_transforms
        )

    def it_replaces_the_element_transforms_to_help_with_subvar_id_keys(
        self, _subvar_aliases_prop_, _subvar_ids_prop_
    ):
        _subvar_aliases_prop_.return_value = tuple(("alias1", "alias2", "alias3"))
        _subvar_ids_prop_.return_value = tuple(("xyz", "s2", "abc"))
        element_transforms = {
            "s1": {"transform": "object"},
            "s2": {"another": "transform"},
            "key": "subvar_id",
        }
        shim_ = _ElementIdShim(None, None, None)

        assert shim_._replaced_element_transforms(element_transforms) == {
            "alias2": {"another": "transform"}
        }

    @pytest.mark.parametrize(
        "dim_dict, expected",
        (
            (
                {
                    "type": {
                        "elements": [{"value": {"id": "a"}}, {"value": {"id": 2}}],
                        "subtype": {"class": "enum"},
                    }
                },
                ("a", 2),
            ),
            (
                {
                    "type": {
                        "elements": [{"id": "a"}, {"id": 2}],
                        "subtype": {"class": "variable"},
                    }
                },
                (),  # --- Not structured like a true subvars dim so returns empty
            ),
        ),
    )
    def it_provides_the_subvar_ids_to_help(self, dim_dict, expected):
        shim_ = _ElementIdShim(None, dim_dict, None)
        assert shim_._subvar_ids == expected

    def it_provides_the_subvar_aliases_to_help(self):
        dimension_dict = {
            "type": {
                "elements": [
                    {"id": 1, "value": {"references": {"alias": "x"}}},
                    {"id": 2, "value": {"references": {"alias": "y"}}},
                    {"id": 3},
                ]
            }
        }
        shim_ = _ElementIdShim(None, dimension_dict, None)

        assert shim_._subvar_aliases == tuple(("x", "y", 3))

    @pytest.mark.parametrize(
        "dimension_dict, dimension_type, expected_has_mr_insertion",
        (
            ({"references": {"view": None}}, DT.MR_SUBVAR, False),
            ({"references": {"view": {}}}, DT.MR_SUBVAR, False),
            ({"references": {"view": {"transform": {}}}}, DT.MR_SUBVAR, False),
            (
                {"references": {"view": {"transform": {"insertions": []}}}},
                DT.MR_SUBVAR,
                False,
            ),
            (
                {"references": {"view": {"transform": {"insertions": ["foo"]}}}},
                DT.MR_SUBVAR,
                True,
            ),
            (
                {"references": {"view": {"transform": {"insertions": ["foo"]}}}},
                DT.CAT,
                False,
            ),
        ),
    )
    def it_knows_if_it_has_mr_insertions(
        self, dimension_dict, dimension_type, expected_has_mr_insertion
    ):
        shim = _ElementIdShim(dimension_type, dimension_dict, None)

        has_mr_insertion = shim._has_mr_insertion

        assert has_mr_insertion == expected_has_mr_insertion

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _raw_element_id_prop_(self, request):
        return property_mock(request, _ElementIdShim, "_raw_element_ids")

    @pytest.fixture
    def _replaced_element_transforms_(self, request):
        return method_mock(request, _ElementIdShim, "_replaced_element_transforms")

    @pytest.fixture
    def _replaced_order_element_ids_(self, request):
        return method_mock(request, _ElementIdShim, "_replaced_order_element_ids")

    @pytest.fixture
    def _subvar_aliases_prop_(self, request):
        return property_mock(request, _ElementIdShim, "_subvar_aliases")

    @pytest.fixture
    def _subvar_ids_prop_(self, request):
        return property_mock(request, _ElementIdShim, "_subvar_ids")


class Describe_Element:
    """Unit-test suite for `cr.cube.dimension.Element` object."""

    def it_knows_its_anchor(self):
        element_dict = {"value": {"references": {"anchor": "top"}, "derived": True}}
        element = Element(element_dict, None, None, None)

        anchor = element.anchor

        assert anchor == "top"

    def but_anchor_is_none_if_not_derived(self):
        element_dict = {"value": {"derived": False}}
        element = Element(element_dict, None, None, None)

        anchor = element.anchor

        assert anchor is None

    def it_knows_its_element_id(self):
        element_dict = {"id": 42}
        element = Element(element_dict, None, None, None)

        element_id = element.element_id

        assert element_id == 42

    def it_knows_its_fill_RGB_color_str(self, element_transforms_):
        element_transforms_.fill = [255, 255, 248]
        element = Element(None, None, element_transforms_, None)

        rgb_color_fill = element.fill

        assert rgb_color_fill == [255, 255, 248]

    def it_knows_its_position_among_all_the_dimension_elements(self):
        element = Element(None, 17, None, None)
        index = element.index
        assert index == 17

    @pytest.mark.parametrize(
        ("element_dict", "transform_name", "expected_value", "fmt_calls"),
        (
            ({}, None, "", ()),
            ({}, "Garf", "Garf", ()),
            ({"name": ""}, None, "", ()),
            ({"name": None}, None, "", ()),
            ({"name": "Bob"}, None, "Bob", ()),
            ({"name": "Hinzufägen"}, None, "Hinzufägen", ()),
            ({"value": ["A", "F"]}, None, "A-F", ("A", "F")),
            ({"value": [1.2, 3.4]}, None, "1.2-3.4", (1.2, 3.4)),
            ({"value": 42}, None, "42", (42,)),
            ({"value": 4.2}, None, "4.2", (4.2,)),
            ({"value": "Bill"}, None, "Bill", ("Bill",)),
            ({"value": "Fähig"}, None, "Fähig", ("Fähig",)),
            ({"value": {"references": {}}}, None, "", ()),
            ({"value": {"references": {"name": "Tom"}}}, None, "Tom", ()),
            ({"value": {"references": {"name": "Tom"}}}, "Harry", "Harry", ()),
            ({"value": {"references": {"name": "Tom"}}}, "", "", ()),
        ),
    )
    def it_knows_its_label(
        self,
        request,
        element_dict,
        transform_name,
        expected_value,
        element_transforms_,
        fmt_calls,
    ):
        element_transforms_.name = transform_name
        element = Element(element_dict, None, element_transforms_, str)
        assert element.label == expected_value

    @pytest.mark.parametrize(
        ("hide", "expected_value"), ((True, True), (False, False), (None, False))
    )
    def it_knows_whether_it_is_explicitly_hidden(self, request, hide, expected_value):
        element_transforms_ = instance_mock(request, _ElementTransforms, hide=hide)
        element = Element(None, None, element_transforms_, None)

        is_hidden = element.is_hidden

        assert is_hidden is expected_value

    @pytest.mark.parametrize(
        ("element_dict", "expected_value"),
        (
            ({}, False),
            ({"missing": None}, False),
            ({"missing": False}, False),
            ({"missing": True}, True),
            # ---not expected values, but just in case---
            ({"missing": 0}, False),
            ({"missing": 1}, True),
        ),
    )
    def it_knows_whether_its_missing_or_valid(self, element_dict, expected_value):
        element = Element(element_dict, None, None, None)

        missing = element.missing

        # ---only True or False, no Truthy or Falsy (so use `is` not `==`)---
        assert missing is expected_value

    @pytest.mark.parametrize(
        ("element_dict", "expected_value"),
        (
            ({}, np.nan),
            ({"numeric_value": None}, np.nan),
            ({"numeric_value": 0}, 0),
            ({"numeric_value": 7}, 7),
            ({"numeric_value": -3.2}, -3.2),
            # ---not expected values, just to document the behavior that
            # ---no attempt is made to convert values to numeric
            ({"numeric_value": "666"}, "666"),
            ({"numeric_value": {}}, {}),
            ({"numeric_value": {"?": 8}}, {"?": 8}),
        ),
    )
    def it_knows_its_numeric_value(self, element_dict, expected_value):
        element = Element(element_dict, None, None, None)

        numeric_value = element.numeric_value

        # ---np.nan != np.nan, but np.nan in [np.nan] works---
        assert numeric_value in [expected_value]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def element_transforms_(self, request):
        return instance_mock(request, _ElementTransforms)


class DescribeElementTransforms:
    """Unit-test suite for `cr.cube.dimension._ElementTransforms` object."""

    @pytest.mark.parametrize(
        "element_transforms_dict, expected_value",
        (
            ({"name": "Zillow", "fill": "#316395"}, "#316395"),
            ({"name": "Zillow", "fill": None}, None),
            ({"name": "Zillow", "fill": [255, 255, 0]}, [255, 255, 0]),
        ),
    )
    def it_knows_its_fill_color_value(self, element_transforms_dict, expected_value):
        element_transforms = _ElementTransforms(element_transforms_dict)

        fill_color_value = element_transforms.fill

        assert fill_color_value == expected_value

    @pytest.mark.parametrize(
        "element_transforms_dict, expected_value",
        (
            ({"hide": True}, True),
            ({"hide": False}, False),
            ({"hide": None}, None),
            ({"hide": 0}, None),
        ),
    )
    def it_knows_when_it_is_explicitly_hidden(
        self, element_transforms_dict, expected_value
    ):
        element_transforms = _ElementTransforms(element_transforms_dict)

        is_hidden = element_transforms.hide

        assert is_hidden is expected_value

    @pytest.mark.parametrize(
        "element_transforms_dict, expected_value",
        (
            ({"name": "MyDisplayName"}, "MyDisplayName"),
            ({"name": ""}, ""),
            ({"name": None}, ""),
            ({"foo": "foo"}, None),
        ),
    )
    def it_knows_its_name(self, element_transforms_dict, expected_value):
        element_transforms = _ElementTransforms(element_transforms_dict)

        name = element_transforms.name

        assert name == expected_value


class Describe_OrderSpec:
    """Unit-test suite for `cr.cube.dimension._OrderSpec` object."""

    @pytest.mark.parametrize(
        "order_dict, expected_value",
        (
            ({}, ()),
            ({"fixed": {}}, ()),
            ({"fixed": {"top": [1, 2]}}, ()),
            ({"fixed": {"bottom": []}}, ()),
            ({"fixed": {"bottom": [3, 4]}}, (3, 4)),
            ({"fixed": {"top": [1, 2], "bottom": [3, 4]}}, (3, 4)),
        ),
    )
    def it_knows_its_bottom_fixed_ids(
        self, _order_dict_prop_, order_dict, expected_value
    ):
        _order_dict_prop_.return_value = order_dict
        assert _OrderSpec(None, None).bottom_fixed_ids == expected_value

    @pytest.mark.parametrize(
        "order_dict, expected_value",
        (
            ({}, CM.PAYLOAD_ORDER),
            ({"type": "explicit"}, CM.EXPLICIT_ORDER),
            ({"type": "payload_order"}, CM.PAYLOAD_ORDER),
        ),
    )
    def it_knows_its_collation_method(
        self, _order_dict_prop_, order_dict, expected_value
    ):
        _order_dict_prop_.return_value = order_dict
        assert _OrderSpec(None, None).collation_method == expected_value

    @pytest.mark.parametrize(
        "order_dict, expected_value",
        (
            ({}, True),
            ({"direction": "descending"}, True),
            ({"direction": "foobar"}, True),
            ({"direction": "ascending"}, False),
        ),
    )
    def it_knows_whether_the_sort_direction_is_descending(
        self, _order_dict_prop_, order_dict, expected_value
    ):
        _order_dict_prop_.return_value = order_dict
        assert _OrderSpec(None, None).descending == expected_value

    def it_knows_the_sort_vector_element_id(self, _order_dict_prop_):
        _order_dict_prop_.return_value = {"element_id": 42}
        assert _OrderSpec(None, None).element_id == 42

    def but_it_raises_when_element_id_not_present(self, _order_dict_prop_):
        _order_dict_prop_.return_value = {}
        with pytest.raises(KeyError) as e:
            _OrderSpec(None, None).element_id
        assert str(e.value) == "'element_id'"

    @pytest.mark.parametrize(
        "order_dict, expected_value",
        (
            ({}, ()),
            ({"element_ids": None}, ()),
            ({"element_ids": []}, ()),
            ({"element_ids": [4, 2]}, (4, 2)),
        ),
    )
    def it_knows_the_ordered_element_ids_for_an_explicit_sort(
        self, _order_dict_prop_, order_dict, expected_value
    ):
        _order_dict_prop_.return_value = order_dict
        assert _OrderSpec(None, None).element_ids == expected_value

    def it_knows_the_sort_vector_insertion_id(self, _order_dict_prop_):
        _order_dict_prop_.return_value = {"insertion_id": 42}
        assert _OrderSpec(None, None).insertion_id == 42

    def but_it_raises_when_insertion_id_not_present(self, _order_dict_prop_):
        _order_dict_prop_.return_value = {}
        with pytest.raises(KeyError) as e:
            _OrderSpec(None, None).insertion_id
        assert str(e.value) == "'insertion_id'"

    def it_knows_the_marginal_specified_as_the_sort_basis(self, _order_dict_prop_):
        _order_dict_prop_.return_value = {"marginal": "scale_mean"}
        assert _OrderSpec(None, None).marginal == MARGINAL.SCALE_MEAN

    def it_knows_the_measure_specified_as_the_sort_basis(self, _order_dict_prop_):
        _order_dict_prop_.return_value = {"measure": "col_percent"}
        assert _OrderSpec(None, None).measure == MEASURE.COLUMN_PERCENT

    def but_it_raises_when_measure_field_is_not_present(self, _order_dict_prop_):
        _order_dict_prop_.return_value = {}
        with pytest.raises(KeyError) as e:
            _OrderSpec(None, None).measure
        assert str(e.value) == "'measure'"

    def and_it_raises_when_the_measure_keyword_is_not_recognized(
        self, _order_dict_prop_
    ):
        _order_dict_prop_.return_value = {"measure": "foobar"}
        with pytest.raises(ValueError) as e:
            _OrderSpec(None, None).measure
        # --- `.endswith()` to accommodate a Python 2.7 vs. 3.0 error message difference
        # --- in Enum module
        assert str(e.value).endswith(" is not a valid MEASURE")

    @pytest.mark.parametrize(
        "order_dict, expected_value",
        (
            ({}, ()),
            ({"fixed": {}}, ()),
            ({"fixed": {"bottom": [1, 2]}}, ()),
            ({"fixed": {"top": []}}, ()),
            ({"fixed": {"top": [3, 4]}}, (3, 4)),
            ({"fixed": {"bottom": [1, 2], "top": [3, 4]}}, (3, 4)),
        ),
    )
    def it_knows_its_top_fixed_ids(self, _order_dict_prop_, order_dict, expected_value):
        _order_dict_prop_.return_value = order_dict
        assert _OrderSpec(None, None).top_fixed_ids == expected_value

    @pytest.mark.parametrize(
        "dim_transforms, expected_value",
        (
            ({}, {}),
            ({"order": None}, {}),
            ({"order": {}}, {}),
            ({"order": {"type": "explicit"}}, {"type": "explicit"}),
        ),
    )
    def it_provides_access_to_the_order_dict_to_help(
        self, dim_transforms, expected_value
    ):
        assert _OrderSpec(None, dim_transforms)._order_dict == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _order_dict_prop_(self, request):
        return property_mock(request, _OrderSpec, "_order_dict")


class Describe_Subtotals:
    """Unit-test suite for `cr.cube.dimension._Subtotals` object."""

    def it_has_sequence_behaviors(self, request, _subtotals_prop_):
        _subtotals_prop_.return_value = (1, 2, 3)
        subtotals = _Subtotals(None, None)

        assert subtotals[1] == 2
        assert subtotals[1:3] == (2, 3)
        assert len(subtotals) == 3
        assert list(n for n in subtotals) == [1, 2, 3]

    def it_provides_the_element_ids_as_a_set_to_help(self, request, valid_elements_):
        valid_elements_.element_ids = tuple(range(3))
        subtotals = _Subtotals(None, valid_elements_)

        element_ids = subtotals._element_ids

        assert element_ids == {0, 1, 2}

    @pytest.mark.parametrize(
        "insertion_dicts, element_ids, expected_value",
        (
            ([], (), ()),
            (["not-a-dict", None], (), ()),
            ([{"function": "hyperdrive"}], (), ()),
            ([{"function": "subtotal", "hide": True}], (), ()),
            ([{"function": "subtotal", "arghhs": []}], (), ()),
            ([{"function": "subtotal", "anchor": 9, "name": "no args"}], (), ()),
            ([{"function": "subtotal", "anchor": 9, "args": [1, 2]}], (), ()),
            (
                [{"function": "subtotal", "anchor": 9, "args": [1, 2], "name": "A"}],
                {3, 4},
                (),
            ),
            (
                [
                    {"function": "subtotal", "anchor": 9, "args": [1, 2], "name": "C"},
                    {"function": "subtotal", "anchor": 9, "args": [3, 4], "name": "Z"},
                    {"function": "subtotal", "anchor": 9, "args": [5, 6], "name": "D"},
                ],
                {1, 2, 5, 8, -1},
                (
                    {"function": "subtotal", "anchor": 9, "args": [1, 2], "name": "C"},
                    {"function": "subtotal", "anchor": 9, "args": [5, 6], "name": "D"},
                ),
            ),
        ),
    )
    def it_removes_invalid_when_iterating_the_valid_subtotal_insertion_dicts_to_help(
        self, _element_ids_prop_, insertion_dicts, element_ids, expected_value
    ):
        _element_ids_prop_.return_value = element_ids
        subtotals = _Subtotals(insertion_dicts, None)

        subtotal_dicts = tuple(subtotals._iter_valid_subtotal_dicts())

        assert subtotal_dicts == expected_value

    def but_it_accepts_valid_insertion_dicts(self, _element_ids_prop_):
        _element_ids_prop_.return_value = {1, 2, 5, 8, -1}
        insertion_dicts = [
            {
                "function": "subtotal",
                "anchor": 9,
                "args": [1],
                "name": "old style args",
            },
            {
                "function": "subtotal",
                "anchor": 9,
                "args": [1],
                "kwargs": {"negative": [2]},
                "name": "old style args with negative",
            },
            {
                "function": "subtotal",
                "anchor": 9,
                "kwargs": {"negative": [2]},
                "name": "negative only",
            },
            {
                "function": "subtotal",
                "anchor": 9,
                "kwargs": {"positive": [1]},
                "name": "new style kwargs",
            },
            {
                "function": "subtotal",
                "anchor": 9,
                "kwargs": {"positive": [1], "negative": [2]},
                "name": "new style kwargs with negative",
            },
        ]
        subtotals = _Subtotals(insertion_dicts, None)

        subtotal_dicts = tuple(subtotals._iter_valid_subtotal_dicts())

        assert subtotal_dicts == tuple(insertion_dicts)

    def it_constructs_its_subtotal_objects_to_help(
        self, request, _iter_valid_subtotal_dicts_, valid_elements_, _Subtotal_
    ):
        subtotal_dicts_ = tuple(
            {"subtotal-dict": idx, "anchor": "top", "id": idx + 1} for idx in range(3)
        )
        subtotal_objs_ = tuple(
            instance_mock(request, _Subtotal, name="subtotal-%d" % idx)
            for idx in range(3)
        )
        _iter_valid_subtotal_dicts_.return_value = iter(subtotal_dicts_)
        _Subtotal_.side_effect = iter(subtotal_objs_)
        subtotals = _Subtotals(None, valid_elements_)

        subtotal_objs = subtotals._subtotals

        assert _Subtotal_.call_args_list == [
            call(subtotal_dicts_[0], valid_elements_),
            call(subtotal_dicts_[1], valid_elements_),
            call(subtotal_dicts_[2], valid_elements_),
        ]
        assert subtotal_objs == subtotal_objs_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _element_ids_prop_(self, request):
        return property_mock(request, _Subtotals, "_element_ids")

    @pytest.fixture
    def _iter_valid_subtotal_dicts_(self, request):
        return method_mock(request, _Subtotals, "_iter_valid_subtotal_dicts")

    @pytest.fixture
    def _Subtotal_(self, request):
        return class_mock(request, "cr.cube.dimension._Subtotal")

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, _Subtotals, "_subtotals")

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, Elements)


class Describe_Subtotal:
    """Unit-test suite for `cr.cube.dimension._Subtotal` object."""

    @pytest.mark.parametrize(
        "subtotal_dict, addend_ids, expected_value",
        (
            ({"args": []}, (), []),
            ({"args": [1, 2]}, (1, 2), [0, 1]),
            ({"args": [0, 2]}, (1, 2), [0, 1]),
            ({"args": [3, 4]}, (3, 4), [2, 3]),
            ({"args": [4, 99]}, (4, 99), [3, 4]),
        ),
    )
    def it_knows_the_addend_idxs(
        self,
        addend_ids_,
        all_elements_,
        valid_elements_,
        subtotal_dict,
        addend_ids,
        expected_value,
    ):
        addend_ids_.return_value = addend_ids
        valid_elements = Elements(
            [
                Element({"id": 1}, None, None, None),
                Element({"id": 2}, None, None, None),
                Element({"id": 3}, None, None, None),
                Element({"id": 4}, None, None, None),
                Element({"id": 99}, None, None, None),
            ]
        )
        subtotal = _Subtotal(subtotal_dict, valid_elements)

        addend_idxs = subtotal.addend_idxs

        np.testing.assert_array_almost_equal(addend_idxs, expected_value)

    @pytest.mark.parametrize(
        "subtotal_dict, element_ids, expected_value",
        (
            ({"anchor": 1}, {1, 2, 3}, 1),
            ({"anchor": 4}, {1, 2, 3}, "bottom"),
            ({"anchor": "Top"}, {1, 2, 3}, "top"),
            # For an undefined anchor default to "bottom"
            ({"anchor": None}, {1, 2, 3}, "bottom"),
        ),
    )
    def it_knows_the_insertion_anchor(
        self, subtotal_dict, element_ids, expected_value, valid_elements_
    ):
        valid_elements_.element_ids = element_ids
        subtotal = _Subtotal(subtotal_dict, valid_elements_)

        assert subtotal.anchor == expected_value

    @pytest.mark.parametrize(
        "subtotal_dict, element_ids, expected_value",
        (
            ({}, {}, ()),
            ({"args": [1]}, {1}, (1,)),
            ({"args": [1, 2, 3]}, {1, 2, 3}, (1, 2, 3)),
            ({"args": [1, 2, 3]}, {1, 3}, (1, 3)),
            ({"args": [3, 2]}, {1, 2, 3}, (3, 2)),
            ({"args": []}, {1, 2, 3}, ()),
            ({"args": [1, 2, 3]}, {}, ()),
            ({"kwargs": {"positive": [1, 2, 3]}}, {1, 2, 3}, (1, 2, 3)),
            ({"args": [1], "kwargs": {"positive": [2]}}, {1, 2, 3}, (2,)),
        ),
    )
    def it_provides_access_to_the_addend_element_ids(
        self, subtotal_dict, element_ids, expected_value, valid_elements_
    ):
        valid_elements_.element_ids = element_ids
        subtotal = _Subtotal(subtotal_dict, valid_elements_)

        assert subtotal.addend_ids == expected_value

    def it_knows_its_insertion_id_when_it_has_been_assigned_one(self):
        assert _Subtotal({"id": 42}, None).insertion_id == 42

    @pytest.mark.parametrize(
        "subtotal_dict, expected_value",
        (({}, ""), ({"name": None}, ""), ({"name": ""}, ""), ({"name": "Joe"}, "Joe")),
    )
    def it_knows_the_subtotal_label(self, subtotal_dict, expected_value):
        assert _Subtotal(subtotal_dict, None).label == expected_value

    @pytest.mark.parametrize(
        "subtotal_dict, element_ids, expected_value",
        (
            ({}, {}, ()),
            ({"kwargs": {"negative": [1]}}, {1}, (1,)),
            ({"kwargs": {"negative": [1, 2, 3]}}, {1, 2, 3}, (1, 2, 3)),
            ({"kwargs": {"negative": [1, 2, 3]}}, {1, 3}, (1, 3)),
            ({"kwargs": {"negative": [3, 2]}}, {1, 2, 3}, (3, 2)),
            ({"kwargs": {"negative": []}}, {1, 2, 3}, ()),
            ({"kwargs": {"negative": [1, 2, 3]}}, {}, ()),
        ),
    )
    def it_provides_access_to_the_subtrahend_element_ids(
        self, subtotal_dict, element_ids, expected_value, valid_elements_
    ):
        valid_elements_.element_ids = element_ids
        subtotal = _Subtotal(subtotal_dict, valid_elements_)

        assert subtotal.subtrahend_ids == expected_value

    @pytest.mark.parametrize(
        "subtotal_dict, subtrahend_ids, expected_value",
        (
            ({"kwargs": {"negative": []}}, (), []),
            ({"kwargs": {"negative": [1, 2]}}, (1, 2), [0, 1]),
            ({"kwargs": {"negative": [0, 2]}}, (1, 2), [0, 1]),
            ({"kwargs": {"negative": [3, 4]}}, (3, 4), [2, 3]),
            ({"kwargs": {"negative": [4, 99]}}, (4, 99), [3, 4]),
        ),
    )
    def it_knows_the_subtrahend_idxs(
        self,
        subtrahend_ids_,
        all_elements_,
        valid_elements_,
        subtotal_dict,
        subtrahend_ids,
        expected_value,
    ):
        subtrahend_ids_.return_value = subtrahend_ids
        valid_elements = Elements(
            [
                Element({"id": 1}, None, None, None),
                Element({"id": 2}, None, None, None),
                Element({"id": 3}, None, None, None),
                Element({"id": 4}, None, None, None),
                Element({"id": 99}, None, None, None),
            ]
        )
        subtotal = _Subtotal(subtotal_dict, valid_elements)

        subtrahend_idxs = subtotal.subtrahend_idxs

        np.testing.assert_array_almost_equal(subtrahend_idxs, expected_value)

    @pytest.mark.parametrize(
        "subtotal_dict, expected_fill",
        (({"fill": "fake_fill"}, "fake_fill"), ({}, None)),
    )
    def it_knows_its_fill(self, subtotal_dict, expected_fill):
        subtotal = _Subtotal(subtotal_dict, None)

        fill = subtotal.fill

        assert fill == expected_fill

    # fixture components ---------------------------------------------

    @pytest.fixture
    def addend_ids_(self, request):
        return property_mock(request, _Subtotal, "addend_ids")

    @pytest.fixture
    def all_elements_(self, request):
        return instance_mock(request, Elements)

    @pytest.fixture
    def subtrahend_ids_(self, request):
        return property_mock(request, _Subtotal, "subtrahend_ids")

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, Elements)
