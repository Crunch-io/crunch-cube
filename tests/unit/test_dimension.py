# encoding: utf-8

"""Unit test suite for cr.cube.dimension module."""

import numpy as np
import pytest

from cr.cube.dimension import (
    AllDimensions,
    _AllElements,
    _ApparentDimensions,
    _BaseDimensions,
    _BaseElements,
    Dimension,
    _DimensionFactory,
    _Element,
    _ElementTransforms,
    _RawDimension,
    _Subtotal,
    _Subtotals,
    _ValidElements,
)
from cr.cube.enums import COLLATION_METHOD as CM, DIMENSION_TYPE as DT

from ..unitutil import (
    ANY,
    call,
    class_mock,
    initializer_mock,
    instance_mock,
    method_mock,
    property_mock,
)


class Describe_BaseDimensions(object):
    def it_has_sequence_behaviors(self, _dimensions_prop_):
        _dimensions_prop_.return_value = (0, 1, 2)
        base_dimensions = _BaseDimensions()

        assert base_dimensions[1] == 1
        assert base_dimensions[1:3] == (1, 2)
        assert len(base_dimensions) == 3
        assert list(n for n in base_dimensions) == [0, 1, 2]

    def it_stores_its_dimension_objects_in_a_tuple_to_help(self):
        base_dimensions = _BaseDimensions()
        with pytest.raises(NotImplementedError):
            base_dimensions._dimensions

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _dimensions_prop_(self, request):
        return property_mock(request, _BaseDimensions, "_dimensions")


class DescribeAllDimensions(object):
    """Unit-test suite for `cr.cube.dimension.AllDimensions` object."""

    def it_provides_access_to_its_ApparentDimensions(
        self,
        _ApparentDimensions_,
        apparent_dimensions_,
        _dimensions_prop_,
        all_dimensions_,
    ):
        _dimensions_prop_.return_value = all_dimensions_
        _ApparentDimensions_.return_value = apparent_dimensions_
        all_dimensions = AllDimensions(None)

        apparent_dimensions = all_dimensions.apparent_dimensions

        _ApparentDimensions_.assert_called_once_with(all_dimensions=all_dimensions_)
        assert apparent_dimensions is apparent_dimensions_

    def it_knows_its_shape(self, request, _dimensions_prop_):
        _dimensions_prop_.return_value = tuple(
            instance_mock(request, Dimension, name="dim-%d" % idx, shape=element_count)
            for idx, element_count in enumerate((3, 2, 1))
        )
        all_dimensions = AllDimensions(None)

        shape = all_dimensions.shape

        assert shape == (3, 2, 1)

    def it_stores_its_dimensions_in_a_tuple_to_help(self, request, _DimensionFactory_):
        dimension_dicts_ = [{"d": 0}, {"d": 1}, {"d": 2}]
        dimensions_ = tuple(
            instance_mock(request, Dimension, name="dim-%d" % idx) for idx in range(3)
        )
        _DimensionFactory_.iter_dimensions.return_value = iter(dimensions_)
        all_dimensions = AllDimensions(dimension_dicts_)

        dimensions = all_dimensions._dimensions

        _DimensionFactory_.iter_dimensions.assert_called_once_with(dimension_dicts_)
        assert dimensions == dimensions_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def all_dimensions_(self, request):
        return instance_mock(request, AllDimensions)

    @pytest.fixture
    def _ApparentDimensions_(self, request):
        return class_mock(request, "cr.cube.dimension._ApparentDimensions")

    @pytest.fixture
    def apparent_dimensions_(self, request):
        return instance_mock(request, _ApparentDimensions)

    @pytest.fixture
    def _DimensionFactory_(self, request):
        return class_mock(request, "cr.cube.dimension._DimensionFactory")

    @pytest.fixture
    def _dimensions_prop_(self, request):
        return property_mock(request, AllDimensions, "_dimensions")


class Describe_ApparentDimensions(object):
    def it_stores_its_dimensions_in_a_tuple_to_help(self, request):
        all_dimensions_ = tuple(
            instance_mock(request, Dimension, name="dim-%d" % idx, dimension_type=dt)
            for idx, dt in enumerate((DT.CAT, DT.MR, DT.MR_CAT))
        )
        apparent_dimensions = _ApparentDimensions(all_dimensions_)

        dimensions = apparent_dimensions._dimensions

        assert dimensions == all_dimensions_[:2]


class Describe_DimensionFactory(object):
    def it_provides_an_interface_classmethod(
        self, request, dimension_dicts_, _init_, _iter_dimensions_
    ):
        dimensions_ = tuple(
            instance_mock(request, Dimension, name="dim-%d" % idx) for idx in range(3)
        )
        _iter_dimensions_.return_value = iter(dimensions_)

        dimension_iter = _DimensionFactory.iter_dimensions(dimension_dicts_)

        # ---ANY is for dimension_factory object (self), which we can't see---
        _init_.assert_called_once_with(ANY, dimension_dicts_)
        _iter_dimensions_.assert_called_once_with(ANY)
        assert tuple(dimension_iter) == dimensions_

    def it_generates_the_dimensions_for_a_cube(
        self, request, dimension_dicts_, _raw_dimensions_prop_, Dimension_
    ):
        dimension_types_ = (DT.CAT, DT.CA_SUBVAR, DT.CA_CAT)
        _raw_dimensions_prop_.return_value = tuple(
            instance_mock(
                request,
                _RawDimension,
                name="raw-dim-%d" % idx,
                dimension_dict=dimension_dicts_[idx],
                dimension_type=dimension_types_[idx],
            )
            for idx in range(3)
        )
        dimensions_ = tuple(
            instance_mock(request, Dimension, name="dim-%d" % idx) for idx in range(3)
        )
        Dimension_.side_effect = iter(dimensions_)
        dimension_factory = _DimensionFactory(None)

        dimension_iter = dimension_factory._iter_dimensions()

        # ---exercising the iterator needs to come first---
        assert tuple(dimension_iter) == dimensions_
        assert Dimension_.call_args_list == [
            call(dimension_dicts_[0], dimension_types_[0]),
            call(dimension_dicts_[1], dimension_types_[1]),
            call(dimension_dicts_[2], dimension_types_[2]),
        ]

    def it_constructs_RawDimension_objects_to_help(
        self, request, dimension_dicts_, _RawDimension_
    ):
        raw_dimensions_ = tuple(
            instance_mock(request, _RawDimension, name="raw-dim-%d" % idx)
            for idx in range(3)
        )
        _RawDimension_.side_effect = iter(raw_dimensions_)
        dimension_factory = _DimensionFactory(dimension_dicts_)

        raw_dimensions = dimension_factory._raw_dimensions

        assert _RawDimension_.call_args_list == [
            call(dimension_dicts_[0], dimension_dicts_),
            call(dimension_dicts_[1], dimension_dicts_),
            call(dimension_dicts_[2], dimension_dicts_),
        ]
        assert raw_dimensions == raw_dimensions_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def Dimension_(self, request):
        return class_mock(request, "cr.cube.dimension.Dimension")

    @pytest.fixture
    def dimension_dicts_(self, request):
        """Tuple of three shallow dimension-dict placeholders."""
        return ({"d": 0}, {"d": 1}, {"d": 2})

    @pytest.fixture
    def _init_(self, request):
        return initializer_mock(request, _DimensionFactory)

    @pytest.fixture
    def _iter_dimensions_(self, request):
        return method_mock(request, _DimensionFactory, "_iter_dimensions")

    @pytest.fixture
    def _RawDimension_(self, request):
        return class_mock(request, "cr.cube.dimension._RawDimension")

    @pytest.fixture
    def _raw_dimensions_prop_(self, request):
        return property_mock(request, _DimensionFactory, "_raw_dimensions")


class Describe_RawDimension(object):
    def it_provides_access_to_the_dimension_dict(self):
        dimension_dict_ = {"dimension": "dict"}
        raw_dimension = _RawDimension(dimension_dict_, None)

        dimension_dict = raw_dimension.dimension_dict

        assert dimension_dict == dimension_dict_

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (
            ({"type": {"class": "categorical"}}, "categorical"),
            (
                {"type": {"class": "enum", "subtype": {"class": "variable"}}},
                "enum.variable",
            ),
        ),
    )
    def it_parses_the_base_type_to_help(self, dimension_dict, expected_value):
        raw_dimension = _RawDimension(dimension_dict, None)

        base_type = raw_dimension._base_type

        assert base_type == expected_value

    def but_it_raises_on_unrecognized_type_class(self):
        raw_dimension = _RawDimension({"type": {"class": "crunched"}}, None)
        with pytest.raises(NotImplementedError):
            raw_dimension._base_type

    @pytest.mark.parametrize(
        "base_type, cat_type, arr_type, expected_value",
        (
            ("categorical", DT.CAT, None, DT.CAT),
            ("enum.variable", None, DT.MR, DT.MR),
            ("enum.datetime", None, None, DT.DATETIME),
            ("enum.numeric", None, None, DT.BINNED_NUMERIC),
            ("enum.text", None, None, DT.TEXT),
        ),
    )
    def it_determines_the_dimension_type(
        self,
        base_type,
        cat_type,
        arr_type,
        expected_value,
        _base_type_prop_,
        _resolve_categorical_,
        _resolve_array_type_,
    ):
        _base_type_prop_.return_value = base_type
        _resolve_categorical_.return_value = cat_type if cat_type else None
        _resolve_array_type_.return_value = arr_type if arr_type else None
        raw_dimension = _RawDimension(None, None)
        resolve_cat_calls = [call(raw_dimension)] if cat_type else []
        resolve_arr_calls = [call(raw_dimension)] if arr_type else []

        dimension_type = raw_dimension.dimension_type

        assert _resolve_categorical_.call_args_list == resolve_cat_calls
        assert _resolve_array_type_.call_args_list == resolve_arr_calls
        assert dimension_type == expected_value

    def but_it_raises_on_unrecognized_base_type(self, _base_type_prop_):
        raw_dimension = _RawDimension(None, None)
        _base_type_prop_.return_value = "hyper.dimensional"
        with pytest.raises(NotImplementedError):
            raw_dimension.dimension_type

    def it_knows_the_dimension_variable_identifier_to_help(self):
        dimension_dict = {"references": {"alias": "varski"}}
        raw_dimension = _RawDimension(dimension_dict, None)

        alias = raw_dimension._alias

        assert alias == "varski"

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (
            ({"type": {}}, False),
            ({"type": {"categories": []}}, False),
            ({"type": {"categories": [{}, {}]}}, False),
            ({"type": {"categories": [{"selected": False}, {}]}}, False),
            ({"type": {"categories": [{"selected": True}, {}]}}, True),
        ),
    )
    def it_can_tell_when_a_dimension_has_a_selected_category_to_help(
        self, dimension_dict, expected_value
    ):
        raw_dimension = _RawDimension(dimension_dict, None)

        has_selected_category = raw_dimension._has_selected_category

        assert has_selected_category is expected_value

    @pytest.mark.parametrize(
        "dimension_dict, expected_value",
        (({"references": {}}, False), ({"references": {"subreferences": {}}}, True)),
    )
    def it_distinguishes_an_array_categorical_type_to_help(
        self, dimension_dict, expected_value
    ):
        raw_dimension = _RawDimension(dimension_dict, None)

        is_array_cat = raw_dimension._is_array_cat

        assert is_array_cat == expected_value

    def it_finds_the_subsequent_raw_dimension_to_help(self, request, dimension_dicts_):
        dimension_dict_ = dimension_dicts_[1]
        next_dimension_dict_ = dimension_dicts_[2]
        raw_dimension = _RawDimension(dimension_dict_, dimension_dicts_)
        # --initializer must be mocked after contructing raw_dimension
        # --otherwise it would have no instance variables
        _init_ = initializer_mock(request, _RawDimension)

        next_raw_dimension = raw_dimension._next_raw_dimension

        _init_.assert_called_once_with(
            next_raw_dimension, next_dimension_dict_, dimension_dicts_
        )
        assert type(next_raw_dimension).__name__ == "_RawDimension"

    def but_it_returns_None_for_the_last_dimension(self, dimension_dicts_):
        dimension_dict_ = dimension_dicts_[2]
        raw_dimension = _RawDimension(dimension_dict_, dimension_dicts_)
        next_raw_dimension = raw_dimension._next_raw_dimension
        assert next_raw_dimension is None

    @pytest.mark.parametrize(
        "is_last, base_type, has_sel_cat, alias, expected_value",
        ((False, None, None, None, DT.CA),),
    )
    def it_resolves_an_array_type_to_help(
        self,
        is_last,
        base_type,
        has_sel_cat,
        alias,
        expected_value,
        raw_dimension_,
        _next_raw_dimension_prop_,
    ):
        raw_dimension_._base_type = base_type
        raw_dimension_._has_selected_category = has_sel_cat
        raw_dimension_._alias = alias
        next_raw_dimension = None if is_last else raw_dimension_
        _next_raw_dimension_prop_.return_value = next_raw_dimension
        raw_dimension = _RawDimension(None, None)

        dimension_type = raw_dimension._resolve_array_type()

        assert dimension_type == expected_value

    @pytest.mark.parametrize(
        "is_array_cat, has_selected_cat, is_logical_type, expected_value",
        (
            (False, False, False, DT.CAT),
            (False, True, False, DT.CAT),
            (False, True, True, DT.LOGICAL),
            (True, False, False, DT.CA_CAT),
            (True, True, True, DT.MR_CAT),
            (True, True, False, DT.CA_CAT),
        ),
    )
    def it_resolves_a_categorical_type_to_help(
        self,
        is_array_cat,
        has_selected_cat,
        is_logical_type,
        expected_value,
        _is_array_cat_prop_,
        _has_selected_category_prop_,
        _is_logical_type_prop_,
    ):
        _is_array_cat_prop_.return_value = is_array_cat
        _has_selected_category_prop_.return_value = has_selected_cat
        _is_logical_type_prop_.return_value = is_logical_type
        raw_dimension = _RawDimension(None, None)

        dimension_type = raw_dimension._resolve_categorical()

        assert dimension_type == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _base_type_prop_(self, request):
        return property_mock(request, _RawDimension, "_base_type")

    @pytest.fixture
    def dimension_dicts_(self):
        return ({"dim", 0}, {"dim", 1}, {"dim", 2})

    @pytest.fixture
    def _has_selected_category_prop_(self, request):
        return property_mock(request, _RawDimension, "_has_selected_category")

    @pytest.fixture
    def _is_array_cat_prop_(self, request):
        return property_mock(request, _RawDimension, "_is_array_cat")

    @pytest.fixture
    def _is_logical_type_prop_(self, request):
        return property_mock(request, _RawDimension, "_is_logical_type")

    @pytest.fixture
    def _next_raw_dimension_prop_(self, request):
        return property_mock(request, _RawDimension, "_next_raw_dimension")

    @pytest.fixture
    def raw_dimension_(self, request):
        return instance_mock(request, _RawDimension)

    @pytest.fixture
    def _resolve_array_type_(self, request):
        return method_mock(request, _RawDimension, "_resolve_array_type")

    @pytest.fixture
    def _resolve_categorical_(self, request):
        return method_mock(request, _RawDimension, "_resolve_categorical")


class DescribeDimension(object):
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
        "order_dict, expected_value",
        (
            ({}, CM.PAYLOAD_ORDER),
            ({"type": "explicit"}, CM.EXPLICIT_ORDER),
            ({"type": "payload_order"}, CM.PAYLOAD_ORDER),
        ),
    )
    def it_knows_its_collation_method(self, request, order_dict, expected_value):
        property_mock(request, Dimension, "order_dict", return_value=order_dict)
        assert Dimension(None, None).collation_method == expected_value

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
            instance_mock(request, _Element, element_id=(i + 1)) for i in range(6)
        )
        assert Dimension(None, None).element_ids == (1, 2, 3, 4, 5, 6)

    def it_computes_its_hidden_idxs(self, request, valid_elements_prop_):
        valid_elements_prop_.return_value = (
            instance_mock(request, _Element, is_hidden=bool(i % 2)) for i in range(7)
        )
        assert Dimension(None, None).hidden_idxs == (1, 3, 5)

    def it_knows_the_numeric_values_of_its_elements(
        self, request, valid_elements_prop_
    ):
        valid_elements_prop_.return_value = tuple(
            instance_mock(request, _Element, numeric_value=numeric_value)
            for numeric_value in (1, 2.2, np.nan)
        )
        assert Dimension(None, None).numeric_values == (1, 2.2, np.nan)

    @pytest.mark.parametrize(
        "dimension_transforms, expected_value",
        (
            (None, {}),
            ({}, {}),
            ({"foo": "bar"}, {}),
            ({"order": {"order": "dict"}}, {"order": "dict"}),
        ),
    )
    def it_provides_access_to_the_transforms_order_dict(
        self, dimension_transforms, expected_value
    ):
        assert Dimension(None, None, dimension_transforms).order_dict == expected_value

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
                                    {"insertion": "dict-1"},
                                    {"insertion": "dict-2"},
                                ]
                            }
                        }
                    }
                },
                [{"insertion": "dict-1"}, {"insertion": "dict-2"}],
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

        _Subtotals_.assert_called_once_with(["subtotal", "dicts"], valid_elements_)
        assert subtotals is subtotals_

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _Subtotals_(self, request):
        return class_mock(request, "cr.cube.dimension._Subtotals")

    @pytest.fixture
    def subtotals_(self, request):
        return instance_mock(request, _Subtotals)

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, _ValidElements)

    @pytest.fixture
    def valid_elements_prop_(self, request):
        return property_mock(request, Dimension, "valid_elements")


class Describe_BaseElements(object):
    """Unit-test suite for `cr.cube.dimension._BaseElements` object."""

    def it_has_sequence_behaviors(self, request, _elements_prop_):
        _elements_prop_.return_value = (1, 2, 3)
        elements = _BaseElements()

        assert elements[1] == 2
        assert elements[1:3] == (2, 3)
        assert len(elements) == 3
        assert list(n for n in elements) == [1, 2, 3]

    def it_knows_the_element_ids(self, request, _elements_prop_):
        _elements_prop_.return_value = tuple(
            instance_mock(request, _Element, element_id=n) for n in (1, 2, 5)
        )
        elements = _BaseElements()

        element_ids = elements.element_ids

        assert element_ids == (1, 2, 5)

    def it_knows_the_element_indices(self, request, _elements_prop_):
        _elements_prop_.return_value = tuple(
            instance_mock(request, _Element, index=index) for index in (1, 3, 4)
        )
        elements = _BaseElements()

        element_idxs = elements.element_idxs

        assert element_idxs == (1, 3, 4)

    def it_can_find_an_element_by_id(self, request, _elements_by_id_prop_):
        elements_ = tuple(
            instance_mock(request, _Element, element_id=element_id)
            for element_id in (3, 7, 11)
        )
        _elements_by_id_prop_.return_value = {
            element_.element_id: element_ for element_ in elements_
        }
        elements = _BaseElements()

        element = elements.get_by_id(7)

        assert element is elements_[1]

    def it_maintains_a_dict_of_elements_by_id_to_help(self, request, _elements_prop_):
        elements_ = tuple(
            instance_mock(request, _Element, element_id=element_id)
            for element_id in (4, 6, 7)
        )
        _elements_prop_.return_value = elements_
        elements = _BaseElements()

        elements_by_id = elements._elements_by_id

        assert elements_by_id == {4: elements_[0], 6: elements_[1], 7: elements_[2]}

    def it_stores_its_elements_in_a_tuple_to_help(self):
        base_elements = _BaseElements()
        # ---must be implemented by each subclass---
        with pytest.raises(NotImplementedError):
            base_elements._elements

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _elements_by_id_prop_(self, request):
        return property_mock(request, _BaseElements, "_elements_by_id")

    @pytest.fixture
    def _elements_prop_(self, request):
        return property_mock(request, _BaseElements, "_elements")


class Describe_AllElements(object):
    """Unit-test suite for `cr.cube.dimension._AllElements` object."""

    def it_provides_access_to_the_ValidElements_object(
        self, request, _elements_prop_, _ValidElements_, valid_elements_
    ):
        elements_ = tuple(
            instance_mock(request, _Element, name="el%s" % idx) for idx in range(3)
        )
        dimension_transforms_dict = {"dimension": "transforms"}
        _elements_prop_.return_value = elements_
        _ValidElements_.return_value = valid_elements_
        all_elements = _AllElements(None, dimension_transforms_dict, None)

        valid_elements = all_elements.valid_elements

        _ValidElements_.assert_called_once_with(elements_, dimension_transforms_dict)
        assert valid_elements is valid_elements_

    def it_creates_its_Element_objects_in_a_local_factory_to_help(
        self,
        request,
        _ElementTransforms_,
        _Element_,
        _iter_element_makings_,
    ):
        element_transforms_ = tuple(
            instance_mock(request, _ElementTransforms, name="element-xfrms-%s" % idx)
            for idx in range(3)
        )
        _ElementTransforms_.side_effect = iter(element_transforms_)
        _iter_element_makings_.return_value = iter(
            (
                (0, {"element": "dict-A"}, {"xfrms": 0}),
                (1, {"element": "dict-B"}, {"xfrms": 1}),
                (2, {"element": "dict-C"}, {"xfrms": 2}),
            )
        )
        elements_ = tuple(
            instance_mock(request, _Element, name="element-%s" % idx)
            for idx in range(3)
        )
        _Element_.side_effect = iter(elements_)
        all_elements = _AllElements(None, None, None)

        elements = all_elements._elements

        assert _ElementTransforms_.call_args_list == [
            call({"xfrms": 0}),
            call({"xfrms": 1}),
            call({"xfrms": 2}),
        ]
        assert _Element_.call_args_list == [
            call({"element": "dict-A"}, 0, element_transforms_[0]),
            call({"element": "dict-B"}, 1, element_transforms_[1]),
            call({"element": "dict-C"}, 2, element_transforms_[2]),
        ]
        assert elements == (elements_[0], elements_[1], elements_[2])

    def it_generates_element_factory_inputs_to_help(self, _element_dicts_prop_):
        dimension_transforms_dict = {
            "elements": {"6": {"element": "xfrms_6"}, "4": {"element": "xfrms_4"}}
        }
        _element_dicts_prop_.return_value = (
            {"id": 4, "element": "dict_4"},
            {"id": 2, "element": "dict_2"},
            {"id": 6, "element": "dict_6"},
        )
        all_elements = _AllElements(None, dimension_transforms_dict, None)

        element_makings = tuple(all_elements._iter_element_makings())

        assert element_makings == (
            (0, {"id": 4, "element": "dict_4"}, {"element": "xfrms_4"}),
            (1, {"id": 2, "element": "dict_2"}, {}),
            (2, {"id": 6, "element": "dict_6"}, {"element": "xfrms_6"}),
        )

    @pytest.mark.parametrize(
        "element_transform, dimension_type, element_dict, expected_value",
        (
            (
                {"S2": {"hide": True}, "key": "subvar_id"},
                DT.NUM_ARRAY,
                {"id": 1, "value": {"references": {"alias": "A1"}, "id": "S2"}},
                "S2",
            ),
            (
                {"A1": {"hide": True}, "key": "alias"},
                DT.NUM_ARRAY,
                {"id": 1, "value": {"references": {"alias": "A1"}, "id": "S2"}},
                "A1",
            ),
            (
                {"1": {"hide": True}},
                DT.CA,
                {"id": 1, "value": {"references": {"alias": "A1"}, "id": "S2"}},
                "S2",
            ),
            (
                {"A1": {"hide": True}, "key": "alias"},
                DT.NUM_ARRAY,
                {"id": 1, "value": {"references": {"alias": ""}, "id": "S2"}},
                1,
            ),
            (
                {"A1": {"hide": True}, "key": "alias"},
                DT.NUM_ARRAY,
                {"id": 1, "value": {"references": {"alias": None}, "id": "S2"}},
                1,
            ),
            ({"1": {"hide": True}}, DT.CAT, {"id": 1, "name": "Male"}, 1),
            ({"0001": {"hide": True}}, DT.CAT, {"id": 1, "name": "Male"}, 1),
        ),
    )
    def it_knows_its_element_id_from_dict(
        self, element_transform, dimension_type, element_dict, expected_value
    ):
        dimension_transforms_dict = {"elements": element_transform}
        all_elements = _AllElements(None, dimension_transforms_dict, dimension_type)

        _element_id_from_dict = all_elements._element_id_from_dict(element_dict)

        assert _element_id_from_dict == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _Element_(self, request):
        return class_mock(request, "cr.cube.dimension._Element")

    @pytest.fixture
    def _element_dicts_prop_(self, request):
        return property_mock(request, _AllElements, "_element_dicts")

    @pytest.fixture
    def _elements_prop_(self, request):
        return property_mock(request, _AllElements, "_elements")

    @pytest.fixture
    def _ElementTransforms_(self, request):
        return class_mock(request, "cr.cube.dimension._ElementTransforms")

    @pytest.fixture
    def _iter_element_makings_(self, request):
        return method_mock(request, _AllElements, "_iter_element_makings")

    @pytest.fixture
    def _ValidElements_(self, request):
        return class_mock(request, "cr.cube.dimension._ValidElements")

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, _ValidElements)


class Describe_ValidElements(object):
    """Unit-test suite for `cr.cube.dimension._ValidElements` object."""

    def it_gets_its_Element_objects_from_an_AllElements_object(
        self, request, all_elements_
    ):
        elements_ = tuple(
            instance_mock(request, _Element, name="element-%s" % idx, missing=missing)
            for idx, missing in enumerate([False, True, False])
        )
        all_elements_.__iter__.return_value = iter(elements_)
        valid_elements = _ValidElements(all_elements_, None)

        elements = valid_elements._elements

        assert elements == (elements_[0], elements_[2])

    @pytest.mark.parametrize(
        "explicit_order, expected_value ",
        ((None, (0, 1, 2)), ((1, 0, 4, 3), (1, 0, 4, 3))),
    )
    def it_knows_its_display_order(
        self, request, all_elements_, explicit_order, expected_value
    ):
        _explicit_order_ = property_mock(request, _ValidElements, "_explicit_order")
        elements_ = tuple(
            instance_mock(request, _Element, missing=False) for _ in range(3)
        )
        all_elements_.__iter__.return_value = iter(elements_)
        valid_elements = _ValidElements(all_elements_, None)
        _explicit_order_.return_value = explicit_order

        display_order = valid_elements.display_order

        assert display_order == expected_value

    @pytest.mark.parametrize(
        "element_transform_dict, expected_value",
        (
            ({}, None),
            ({"order": {"element_ids": [1, 3, 2], "type": "explicit"}}, (0, 2, 1)),
            ({"order": {"element_ids": [-1, 3, 2], "type": "explicit"}}, (2, 1, 0)),
            ({"order": {"element_ids": [1, 2, 3], "type": "explicit"}}, (0, 1, 2)),
        ),
    )
    def it_returns_the_correct_explicit_order(
        self, all_elements_, element_transform_dict, expected_value
    ):
        elements_ = tuple(_Element({"id": idx + 1}, None, None) for idx in range(3))
        all_elements_.__iter__.return_value = iter(elements_)
        valid_elements = _ValidElements(all_elements_, element_transform_dict)

        _explicit_order_ = valid_elements._explicit_order

        assert _explicit_order_ == expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def all_elements_(self, request):
        return instance_mock(request, _AllElements)


class Describe_Element(object):
    """Unit-test suite for `cr.cube.dimension._Element` object."""

    def it_knows_its_element_id(self):
        element_dict = {"id": 42}
        element = _Element(element_dict, None, None)

        element_id = element.element_id

        assert element_id == 42

    def it_knows_its_fill_RGB_color_str(self, element_transforms_):
        element_transforms_.fill = [255, 255, 248]
        element = _Element(None, None, element_transforms_)

        rgb_color_fill = element.fill

        assert rgb_color_fill == [255, 255, 248]

    def it_knows_its_position_among_all_the_dimension_elements(self):
        element = _Element(None, 17, None)
        index = element.index
        assert index == 17

    # TODO: add test cases that exercise element-name transform
    @pytest.mark.parametrize(
        ("element_dict", "expected_value"),
        (
            ({}, ""),
            ({"name": ""}, ""),
            ({"name": None}, ""),
            ({"name": "Bob"}, "Bob"),
            ({"name": "Hinzufägen"}, "Hinzufägen"),
            ({"value": ["A", "F"]}, "A-F"),
            ({"value": [1.2, 3.4]}, "1.2-3.4"),
            ({"value": 42}, "42"),
            ({"value": 4.2}, "4.2"),
            ({"value": "Bill"}, "Bill"),
            ({"value": "Fähig"}, "Fähig"),
            ({"value": {"references": {}}}, ""),
            ({"value": {"references": {"name": "Tom"}}}, "Tom"),
        ),
    )
    def it_knows_its_label(self, element_dict, expected_value, element_transforms_):
        element_transforms_.name = None
        element = _Element(element_dict, None, element_transforms_)

        label = element.label

        assert label == expected_value

    @pytest.mark.parametrize(
        ("hide", "expected_value"), ((True, True), (False, False), (None, False))
    )
    def it_knows_whether_it_is_explicitly_hidden(self, request, hide, expected_value):
        element_transforms_ = instance_mock(request, _ElementTransforms, hide=hide)
        element = _Element(None, None, element_transforms_)

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
        element = _Element(element_dict, None, None)

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
        element = _Element(element_dict, None, None)

        numeric_value = element.numeric_value

        # ---np.nan != np.nan, but np.nan in [np.nan] works---
        assert numeric_value in [expected_value]

    # fixture components ---------------------------------------------

    @pytest.fixture
    def element_transforms_(self, request):
        return instance_mock(request, _ElementTransforms)


class DescribeElementTransforms(object):
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


class Describe_Subtotals(object):
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
        self, insertion_dicts, element_ids, expected_value, _element_ids_prop_
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
        subtotal_dicts_ = tuple({"subtotal-dict": idx} for idx in range(3))
        subtotal_objs_ = tuple(
            instance_mock(request, _Subtotal, name="subtotal-%d" % idx)
            for idx in range(3)
        )
        _iter_valid_subtotal_dicts_.return_value = iter(subtotal_dicts_)
        _Subtotal_.side_effect = iter(subtotal_objs_)
        subtotals = _Subtotals(None, valid_elements_)

        subtotal_objs = subtotals._subtotals

        assert _Subtotal_.call_args_list == [
            call(subtotal_dicts_[0], valid_elements_, 1),
            call(subtotal_dicts_[1], valid_elements_, 2),
            call(subtotal_dicts_[2], valid_elements_, 3),
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
        return instance_mock(request, _ValidElements)


class Describe_Subtotal(object):
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
        elements_ = (
            _Element({"id": 1}, None, None),
            _Element({"id": 2}, None, None),
            _Element({"id": 3}, None, None),
            _Element({"id": 4}, None, None),
            _Element({"id": 99}, None, None),
        )
        all_elements_.__iter__.return_value = iter(elements_)
        valid_elements = _ValidElements(all_elements_, None)
        subtotal = _Subtotal(subtotal_dict, valid_elements, None)

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
        subtotal = _Subtotal(subtotal_dict, valid_elements_, None)

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
        subtotal = _Subtotal(subtotal_dict, valid_elements_, None)

        assert subtotal.addend_ids == expected_value

    def it_knows_its_insertion_id_when_it_has_been_assigned_one(self):
        assert _Subtotal({"insertion_id": 42}, None, None).insertion_id == 42

    def and_it_uses_its_fallback_insertion_id_when_not_assigned_one(self):
        assert _Subtotal({}, None, 24).insertion_id == 24

    @pytest.mark.parametrize(
        "subtotal_dict, expected_value",
        (({}, ""), ({"name": None}, ""), ({"name": ""}, ""), ({"name": "Joe"}, "Joe")),
    )
    def it_knows_the_subtotal_label(self, subtotal_dict, expected_value):
        assert _Subtotal(subtotal_dict, None, None).label == expected_value

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
        subtotal = _Subtotal(subtotal_dict, valid_elements_, None)

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
        elements_ = (
            _Element({"id": 1}, None, None),
            _Element({"id": 2}, None, None),
            _Element({"id": 3}, None, None),
            _Element({"id": 4}, None, None),
            _Element({"id": 99}, None, None),
        )
        all_elements_.__iter__.return_value = iter(elements_)
        valid_elements = _ValidElements(all_elements_, None)
        subtotal = _Subtotal(subtotal_dict, valid_elements, None)

        subtrahend_idxs = subtotal.subtrahend_idxs

        np.testing.assert_array_almost_equal(subtrahend_idxs, expected_value)

    # fixture components ---------------------------------------------

    @pytest.fixture
    def addend_ids_(self, request):
        return property_mock(request, _Subtotal, "addend_ids")

    @pytest.fixture
    def all_elements_(self, request):
        return instance_mock(request, _AllElements)

    @pytest.fixture
    def subtrahend_ids_(self, request):
        return property_mock(request, _Subtotal, "subtrahend_ids")

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, _ValidElements)
