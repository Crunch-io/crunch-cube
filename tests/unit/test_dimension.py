# encoding: utf-8

"""Unit test suite for cr.cube.dimension module."""

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np
import pytest

from cr.cube.dimension import (
    _AllElements, _BaseElement, _BaseElements, _Category, Dimension,
    _Element, _Subtotal, _Subtotals, _ValidElements
)

from ..unitutil import (
    call, class_mock, instance_mock, method_mock, property_mock
)


class DescribeDimension(object):

    def it_knows_its_description(self, description_fixture):
        dimension_dict, expected_value = description_fixture
        dimension = Dimension(dimension_dict)

        description = dimension.description

        assert description == expected_value

    def it_knows_its_dimension_type(self, type_fixture):
        dimension_dict, next_dimension_dict, expected_value = type_fixture
        dimension = Dimension(dimension_dict, next_dimension_dict)

        dimension_type = dimension.dimension_type

        assert dimension_type == expected_value

    def it_provides_subtotal_indices(
            self, hs_indices_fixture, is_selections_prop_, _subtotals_prop_):
        is_selections, subtotals_, expected_value = hs_indices_fixture
        is_selections_prop_.return_value = is_selections
        _subtotals_prop_.return_value = subtotals_
        dimension = Dimension(None, None)

        hs_indices = dimension.hs_indices

        assert hs_indices == expected_value

    def it_knows_the_numeric_values_of_its_elements(
            self, request, _valid_elements_prop_):
        _valid_elements_prop_.return_value = tuple(
            instance_mock(request, _BaseElement, numeric_value=numeric_value)
            for numeric_value in (1, 2.2, np.nan)
        )
        dimension = Dimension(None, None)

        numeric_values = dimension.numeric_values

        assert numeric_values == (1, 2.2, np.nan)

    def it_provides_access_to_its_subtotals_to_help(
            self, subtotals_fixture, _Subtotals_, subtotals_,
            _valid_elements_prop_, valid_elements_):
        dimension_dict, insertion_dicts = subtotals_fixture
        _valid_elements_prop_.return_value = valid_elements_
        _Subtotals_.return_value = subtotals_
        dimension = Dimension(dimension_dict, None)

        subtotals = dimension._subtotals

        _Subtotals_.assert_called_once_with(insertion_dicts, valid_elements_)
        assert subtotals is subtotals_

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        ({'references': {}}, ''),
        ({'references': {'description': None}}, ''),
        ({'references': {'description': ''}}, ''),
        ({'references': {'description': 'Crunchiness'}}, 'Crunchiness'),
    ])
    def description_fixture(self, request):
        dimension_dict, expected_value = request.param
        return dimension_dict, expected_value

    @pytest.fixture(params=[
        (True, ()),
        (False, ((0, (1, 2)), (3, (4, 5)))),
    ])
    def hs_indices_fixture(self, request):
        is_selections, expected_value = request.param
        subtotals_ = (
            instance_mock(
                request, _Subtotal, anchor_idx=idx * 3,
                addend_idxs=(idx + 1 + (2 * idx), idx + 2 + (2 * idx))
            ) for idx in range(2)
        )
        return is_selections, subtotals_, expected_value

    @pytest.fixture(params=[
        ({}, []),
        ({'references': {}}, []),
        ({'references': {'view': {}}}, []),
        ({'references': {'view': {'transform': {}}}}, []),
        ({'references': {'view': {'transform': {'insertions': []}}}}, []),
        ({'references': {'view': {'transform': {'insertions': [
            {'insertion': 'dict-1'},
            {'insertion': 'dict-2'}]}}}},
         [
            {'insertion': 'dict-1'},
            {'insertion': 'dict-2'}])
    ])
    def subtotals_fixture(self, request):
        dimension_dict, insertion_dicts = request.param
        dimension_dict['type'] = {
            'class': 'categorical',
            'categories': [],
        }
        return dimension_dict, insertion_dicts

    # fixture components ---------------------------------------------

    @pytest.fixture
    def is_selections_prop_(self, request):
        return property_mock(request, Dimension, 'is_selections')

    @pytest.fixture
    def _Subtotals_(self, request):
        return class_mock(request, 'cr.cube.dimension._Subtotals')

    @pytest.fixture
    def subtotals_(self, request):
        return instance_mock(request, _Subtotals)

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, Dimension, '_subtotals')

    @pytest.fixture(params=[
        ({'type': {'class': 'categorical'}},
         None, 'categorical'),
        ({'type': {'class': 'enum', 'subtype': {'class': 'datetime'}},
          'references': {}},
         None, 'datetime'),
        ({'type': {'class': 'enum'}, 'references': {'subreferences': []}},
         None, 'categorical_array'),
        ({'type': {'class': 'enum'}, 'references': {'subreferences': []}},
         {'type': {'categories': [{'id': 1}, {'id': 0}, {'id': -1}, ]}},
         'multiple_response'),
        ({'type': {'subtype': {'class': 'numeric'}}},
         None, 'numeric'),
        ({'type': {'subtype': {'class': 'text'}}},
         None, 'text'),
    ])
    def type_fixture(self, request):
        dimension_dict, next_dimension_dict, expected_value = request.param
        return dimension_dict, next_dimension_dict, expected_value

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, _ValidElements)

    @pytest.fixture
    def _valid_elements_prop_(self, request):
        return property_mock(request, Dimension, '_valid_elements')


class Describe_BaseElements(object):

    def it_has_sequence_behaviors(self, request, _elements_prop_):
        _elements_prop_.return_value = (1, 2, 3)
        elements = _BaseElements(None)

        assert elements[1] == 2
        assert elements[1:3] == (2, 3)
        assert len(elements) == 3
        assert list(n for n in elements) == [1, 2, 3]

    def it_knows_the_element_ids(self, request, _elements_prop_):
        _elements_prop_.return_value = tuple(
            instance_mock(request, _BaseElement, element_id=n)
            for n in (1, 2, 5)
        )
        elements = _BaseElements(None)

        element_ids = elements.element_ids

        assert element_ids == (1, 2, 5)

    def it_knows_the_element_indices(self, request, _elements_prop_):
        _elements_prop_.return_value = tuple(
            instance_mock(request, _BaseElement, index=index)
            for index in (1, 3, 4)
        )
        elements = _BaseElements(None)

        element_idxs = elements.element_idxs

        assert element_idxs == (1, 3, 4)

    def it_can_find_an_element_by_id(self, request, _elements_by_id_prop_):
        elements_ = tuple(
            instance_mock(request, _BaseElement, element_id=element_id)
            for element_id in (3, 7, 11)
        )
        _elements_by_id_prop_.return_value = {
            element_.element_id: element_ for element_ in elements_
        }
        elements = _BaseElements(None)

        element = elements.get_by_id(7)

        assert element is elements_[1]

    def it_provides_element_factory_inputs_to_help(self, makings_fixture):
        type_dict, expected_element_class = makings_fixture[:2]
        expected_element_dicts = makings_fixture[2]
        elements = _BaseElements(type_dict)

        ElementCls, element_dicts = elements._element_makings

        assert ElementCls == expected_element_class
        assert element_dicts == expected_element_dicts

    def it_maintains_a_dict_of_elements_by_id_to_help(
            self, request, _elements_prop_):
        elements_ = tuple(
            instance_mock(request, _BaseElement, element_id=element_id)
            for element_id in (4, 6, 7)
        )
        _elements_prop_.return_value = elements_
        elements = _BaseElements(None)

        elements_by_id = elements._elements_by_id

        assert elements_by_id == {
            4: elements_[0],
            6: elements_[1],
            7: elements_[2],
        }

    def it_stores_its_elements_in_a_tuple_to_help(self):
        base_elements = _BaseElements(None)
        # ---must be implemented by each subclass---
        with pytest.raises(NotImplementedError):
            base_elements._elements

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        ({'class': 'categorical', 'categories': ['cat', 'dicts']},
         _Category, ['cat', 'dicts']),
        ({'class': 'enum', 'elements': ['element', 'dicts']},
         _Element, ['element', 'dicts']),
    ])
    def makings_fixture(self, request):
        type_dict, expected_element_cls, expected_element_dicts = request.param
        return type_dict, expected_element_cls, expected_element_dicts

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _elements_by_id_prop_(self, request):
        return property_mock(request, _BaseElements, '_elements_by_id')

    @pytest.fixture
    def _elements_prop_(self, request):
        return property_mock(request, _BaseElements, '_elements')


class Describe_AllElements(object):

    def it_provides_access_to_the_ValidElements_object(
            self, request, _elements_prop_, _ValidElements_, valid_elements_):
        elements_ = tuple(
            instance_mock(request, _BaseElement, name='el%s' % idx)
            for idx in range(3)
        )
        _elements_prop_.return_value = elements_
        _ValidElements_.return_value = valid_elements_
        all_elements = _AllElements(None)

        valid_elements = all_elements.valid_elements

        _ValidElements_.assert_called_once_with(elements_)
        assert valid_elements is valid_elements_

    def it_creates_its_Element_objects_in_its_local_factory(
            self, request, _element_makings_prop_, _BaseElement_):
        element_dicts_ = (
            {'element': 'dict-A'},
            {'element': 'dict-B'},
            {'element': 'dict-C'},
        )
        elements_ = tuple(
            instance_mock(request, _BaseElement, name='element-%s' % idx)
            for idx in range(3)
        )
        _element_makings_prop_.return_value = _BaseElement_, element_dicts_
        _BaseElement_.side_effect = iter(elements_)
        all_elements = _AllElements(None)

        elements = all_elements._elements

        assert _BaseElement_.call_args_list == [
            call({'element': 'dict-A'}, 0),
            call({'element': 'dict-B'}, 1),
            call({'element': 'dict-C'}, 2),
        ]
        assert elements == (elements_[0], elements_[1], elements_[2])

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _BaseElement_(self, request):
        return class_mock(request, 'cr.cube.dimension._BaseElement')

    @pytest.fixture
    def _element_makings_prop_(self, request):
        return property_mock(request, _AllElements, '_element_makings')

    @pytest.fixture
    def _elements_prop_(self, request):
        return property_mock(request, _AllElements, '_elements')

    @pytest.fixture
    def _ValidElements_(self, request):
        return class_mock(request, 'cr.cube.dimension._ValidElements')

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, _ValidElements)


class Describe_ValidElements(object):

    def it_gets_its_Element_objects_from_an_AllElements_object(
            self, request, all_elements_):
        elements_ = tuple(
            instance_mock(
                request, _BaseElement,
                name='element-%s' % idx,
                missing=missing
            )
            for idx, missing in enumerate([False, True, False])
        )
        all_elements_.__iter__.return_value = iter(elements_)
        valid_elements = _ValidElements(all_elements_)

        elements = valid_elements._elements

        assert elements == (elements_[0], elements_[2])

    # fixture components ---------------------------------------------

    @pytest.fixture
    def all_elements_(self, request):
        return instance_mock(request, _AllElements)


class Describe_BaseElement(object):

    def it_knows_its_element_id(self):
        element_dict = {'id': 42}
        element = _BaseElement(element_dict, None)

        element_id = element.element_id

        assert element_id == 42

    def it_knows_its_position_among_all_the_dimension_elements(self):
        element = _BaseElement(None, 17)
        index = element.index
        assert index == 17

    def it_knows_whether_its_missing_or_valid(self, missing_fixture):
        element_dict, expected_value = missing_fixture
        element = _BaseElement(element_dict, None)

        missing = element.missing

        # ---only True or False, no Truthy or Falsy (so use `is` not `==`)---
        assert missing is expected_value

    def it_knows_its_numeric_value(self, numeric_value_fixture):
        element_dict, expected_value = numeric_value_fixture
        element = _BaseElement(element_dict, None)

        numeric_value = element.numeric_value

        # ---np.nan != np.nan, but np.nan in [np.nan] works---
        assert numeric_value in [expected_value]

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        ({}, False),
        ({'missing': None}, False),
        ({'missing': False}, False),
        ({'missing': True}, True),
        # ---not expected values, but just in case---
        ({'missing': 0}, False),
        ({'missing': 1}, True),
    ])
    def missing_fixture(self, request):
        element_dict, expected_value = request.param
        return element_dict, expected_value

    @pytest.fixture(params=[
        ({}, np.nan),
        ({'numeric_value': None}, np.nan),
        ({'numeric_value': 0}, 0),
        ({'numeric_value': 7}, 7),
        ({'numeric_value': -3.2}, -3.2),
        # ---not expected values, just to document the behavior that
        # ---no attempt is made to convert values to numeric
        ({'numeric_value': '666'}, '666'),
        ({'numeric_value': {}}, {}),
        ({'numeric_value': {'?': 8}}, {'?': 8}),
    ])
    def numeric_value_fixture(self, request):
        element_dict, expected_value = request.param
        return element_dict, expected_value


class Describe_Category(object):

    def it_knows_its_label(self, label_fixture):
        category_dict, expected_value = label_fixture
        category = _Category(category_dict, None)

        label = category.label

        assert label == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        ({}, ''),
        ({'name': ''}, ''),
        ({'name': None}, ''),
        ({'name': 'Bob'}, 'Bob'),
        ({'name': 'Hinzufägen'}, 'Hinzufägen'),
    ])
    def label_fixture(self, request):
        category_dict, expected_value = request.param
        return category_dict, expected_value


class Describe_Element(object):

    def it_knows_its_label(self, label_fixture):
        element_dict, expected_value = label_fixture
        element = _Element(element_dict, None)

        label = element.label

        assert label == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        ({}, ''),
        ({'value': ['A', 'F']}, 'A-F'),
        ({'value': [1.2, 3.4]}, '1.2-3.4'),
        ({'value': 42}, '42'),
        ({'value': 4.2}, '4.2'),
        ({'value': 'Bill'}, 'Bill'),
        ({'value': 'Fähig'}, 'Fähig'),
        ({'value': {'references': {}}}, ''),
        ({'value': {'references': {'name': 'Tom'}}}, 'Tom'),
    ])
    def label_fixture(self, request):
        element_dict, expected_value = request.param
        return element_dict, expected_value


class Describe_Subtotals(object):

    def it_has_sequence_behaviors(self, request, _subtotals_prop_):
        _subtotals_prop_.return_value = (1, 2, 3)
        subtotals = _Subtotals(None, None)

        assert subtotals[1] == 2
        assert subtotals[1:3] == (2, 3)
        assert len(subtotals) == 3
        assert list(n for n in subtotals) == [1, 2, 3]

    def it_can_iterate_subtotals_having_a_given_anchor(
            self, request, _subtotals_prop_):
        subtotals_ = tuple(
            instance_mock(
                request, _Subtotal, name='subtotal-%d' % idx, anchor=anchor
            )
            for idx, anchor in enumerate(['bottom', 2, 'bottom'])
        )
        _subtotals_prop_.return_value = subtotals_
        subtotals = _Subtotals(None, None)

        subtotals_with_anchor = tuple(subtotals.iter_for_anchor('bottom'))

        assert subtotals_with_anchor == (subtotals_[0], subtotals[2])

    def it_provides_the_element_ids_as_a_set_to_help(
            self, request, valid_elements_):
        valid_elements_.element_ids = tuple(range(3))
        subtotals = _Subtotals(None, valid_elements_)

        element_ids = subtotals._element_ids

        assert element_ids == {0, 1, 2}

    def it_iterates_the_valid_subtotal_insertion_dicts_to_help(
            self, iter_valid_fixture, _element_ids_prop_):
        insertion_dicts, element_ids, expected_value = iter_valid_fixture
        _element_ids_prop_.return_value = element_ids
        subtotals = _Subtotals(insertion_dicts, None)

        subtotal_dicts = tuple(subtotals._iter_valid_subtotal_dicts())

        assert subtotal_dicts == expected_value

    def it_constructs_its_subtotal_objects_to_help(
            self, request, _iter_valid_subtotal_dicts_, valid_elements_,
            _Subtotal_):
        subtotal_dicts_ = tuple({'subtotal-dict': idx} for idx in range(3))
        subtotal_objs_ = tuple(
            instance_mock(request, _Subtotal, name='subtotal-%d' % idx)
            for idx in range(3)
        )
        _iter_valid_subtotal_dicts_.return_value = iter(subtotal_dicts_)
        _Subtotal_.side_effect = iter(subtotal_objs_)
        subtotals = _Subtotals(None, valid_elements_)

        subtotal_objs = subtotals._subtotals

        assert _Subtotal_.call_args_list == [
            call(subtot_dict_, valid_elements_)
            for subtot_dict_ in subtotal_dicts_
        ]
        assert subtotal_objs == subtotal_objs_

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        ([], (), ()),
        (['not-a-dict', None], (), ()),
        ([{'function': 'hyperdrive'}], (), ()),
        ([{'function': 'subtotal', 'arghhs': []}], (), ()),
        ([{'function': 'subtotal', 'anchor': 9, 'args': [1, 2]}], (), ()),
        ([{'function': 'subtotal', 'anchor': 9, 'args': [1, 2], 'name': 'A'}],
         {3, 4}, ()),
        ([{'function': 'subtotal', 'anchor': 9, 'args': [1, 2], 'name': 'B'}],
         {1, 2, 3, 4, 5, 8, -1},
         ({'function': 'subtotal', 'anchor': 9, 'args': [1, 2], 'name': 'B'},)),
        ([{'function': 'subtotal', 'anchor': 9, 'args': [1, 2], 'name': 'C'},
          {'function': 'subtotal', 'anchor': 9, 'args': [3, 4], 'name': 'Z'},
          {'function': 'subtotal', 'anchor': 9, 'args': [5, 6], 'name': 'D'}],
         {1, 2, 5, 8, -1},
         ({'function': 'subtotal', 'anchor': 9, 'args': [1, 2], 'name': 'C'},
          {'function': 'subtotal', 'anchor': 9, 'args': [5, 6], 'name': 'D'})),
    ])
    def iter_valid_fixture(self, request):
        insertion_dicts, element_ids, expected_value = request.param
        return insertion_dicts, element_ids, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def _element_ids_prop_(self, request):
        return property_mock(request, _Subtotals, '_element_ids')

    @pytest.fixture
    def _iter_valid_subtotal_dicts_(self, request):
        return method_mock(request, _Subtotals, '_iter_valid_subtotal_dicts')

    @pytest.fixture
    def _Subtotal_(self, request):
        return class_mock(request, 'cr.cube.dimension._Subtotal')

    @pytest.fixture
    def _subtotals_prop_(self, request):
        return property_mock(request, _Subtotals, '_subtotals')

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, _ValidElements)


class Describe_Subtotal(object):

    def it_knows_the_insertion_anchor(self, anchor_fixture, valid_elements_):
        subtotal_dict, element_ids, expected_value = anchor_fixture
        valid_elements_.element_ids = element_ids
        subtotal = _Subtotal(subtotal_dict, valid_elements_)

        anchor = subtotal.anchor

        assert anchor == expected_value

    def it_knows_the_index_of_the_anchor_element(
            self, anchor_idx_fixture, anchor_prop_, valid_elements_,
            element_):
        anchor, index, calls, expected_value = anchor_idx_fixture
        anchor_prop_.return_value = anchor
        valid_elements_.get_by_id.return_value = element_
        element_.index = index
        subtotal = _Subtotal(None, valid_elements_)

        anchor_idx = subtotal.anchor_idx

        assert valid_elements_.get_by_id.call_args_list == calls
        assert anchor_idx == expected_value

    def it_provides_access_to_the_addend_element_ids(
            self, addend_ids_fixture, valid_elements_):
        subtotal_dict, element_ids, expected_value = addend_ids_fixture
        valid_elements_.element_ids = element_ids
        subtotal = _Subtotal(subtotal_dict, valid_elements_)

        addend_ids = subtotal.addend_ids

        assert addend_ids == expected_value

    def it_provides_access_to_the_addend_element_indices(
            self, request, addend_ids_prop_, valid_elements_):
        addend_ids_prop_.return_value = (3, 6, 9)
        valid_elements_.get_by_id.side_effect = iter(
            instance_mock(request, _BaseElement, index=index)
            for index in (2, 4, 6)
        )
        subtotal = _Subtotal(None, valid_elements_)

        addend_idxs = subtotal.addend_idxs

        assert valid_elements_.get_by_id.call_args_list == [
            call(3), call(6), call(9)
        ]
        assert addend_idxs == (2, 4, 6)

    def it_knows_the_subtotal_label(self, label_fixture):
        subtotal_dict, expected_value = label_fixture
        subtotal = _Subtotal(subtotal_dict, None)

        label = subtotal.label

        assert label == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        ({}, {}, ()),
        ({'args': [1]}, {1}, (1,)),
        ({'args': [1, 2, 3]}, {1, 2, 3}, (1, 2, 3)),
        ({'args': [1, 2, 3]}, {1, 3}, (1, 3)),
        ({'args': [3, 2]}, {1, 2, 3}, (3, 2)),
        ({'args': []}, {1, 2, 3}, ()),
        ({'args': [1, 2, 3]}, {}, ()),
    ])
    def addend_ids_fixture(self, request):
        subtotal_dict, element_ids, expected_value = request.param
        return subtotal_dict, element_ids, expected_value

    @pytest.fixture(params=[
        ({'anchor': 1}, {1, 2, 3}, 1),
        ({'anchor': 4}, {1, 2, 3}, 'bottom'),
        ({'anchor': 'Top'}, {1, 2, 3}, 'top'),
    ])
    def anchor_fixture(self, request):
        subtotal_dict, element_ids, expected_value = request.param
        return subtotal_dict, element_ids, expected_value

    @pytest.fixture(params=[
        ('top', None, 0, 'top'),
        ('bottom', None, 0, 'bottom'),
        (42, 7, 1, 7),
    ])
    def anchor_idx_fixture(self, request):
        anchor, index, call_count, expected_value = request.param
        calls = [call(anchor)] * call_count
        return anchor, index, calls, expected_value

    @pytest.fixture(params=[
        ({}, ''),
        ({'name': None}, ''),
        ({'name': ''}, ''),
        ({'name': 'Joe'}, 'Joe'),
    ])
    def label_fixture(self, request):
        subtotal_dict, expected_value = request.param
        return subtotal_dict, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def addend_ids_prop_(self, request):
        return property_mock(request, _Subtotal, 'addend_ids')

    @pytest.fixture
    def anchor_prop_(self, request):
        return property_mock(request, _Subtotal, 'anchor')

    @pytest.fixture
    def element_(self, request):
        return instance_mock(request, _BaseElement)

    @pytest.fixture
    def valid_elements_(self, request):
        return instance_mock(request, _ValidElements)
