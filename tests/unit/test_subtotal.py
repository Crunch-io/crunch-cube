from unittest import TestCase
from mock import Mock
from mock import patch
from cr.cube.subtotal import Subtotal


class TestSubtotal(TestCase):

    invalid_subtotal_1 = {
        u'anchor': 0,
        u'name': u'This is respondent ideology',
    }
    invalid_subtotal_2 = {
        u'anchor': 2,
        u'args': [1, 2],
        u'function': u'fake_fcn_not_subtotal',
        u'name': u'Liberal net',
    }
    valid_subtotal = {
        u'anchor': 5,
        u'args': [5, 4],
        u'function': u'subtotal',
        u'name': u'Conservative net',
    }

    valid_subtotal_anchor_bottom = {
        u'anchor': 'Bottom',
        u'args': [5, 4],
        u'function': u'subtotal',
        u'name': u'Conservative net',
    }

    def test_data(self):
        subtotal = Subtotal(self.valid_subtotal, Mock())
        expected = self.valid_subtotal
        actual = subtotal.data
        self.assertEqual(actual, expected)

    def test_is_invalid_when_missing_keys(self):
        subtotal = Subtotal(self.invalid_subtotal_1, Mock())
        expected = False
        actual = subtotal.is_valid
        self.assertEqual(actual, expected)

    def test_is_invalid_when_not_subtotal(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.invalid_subtotal_2, dim)
        expected = False
        actual = subtotal.is_valid
        self.assertEqual(actual, expected)

    def test_is_valid(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.valid_subtotal, dim)
        expected = True
        actual = subtotal.is_valid
        self.assertEqual(actual, expected)

    def test_is_invalid_when_hs_ids_not_in_dim_elements(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 101}, {'id': 102}]
        subtotal = Subtotal(self.valid_subtotal, dim)
        expected = False
        actual = subtotal.is_valid
        self.assertEqual(actual, expected)

    def test_anchor_on_invalid_missing_keys(self):
        subtotal = Subtotal(self.invalid_subtotal_1, Mock())
        expected = None
        actual = subtotal.anchor
        self.assertEqual(actual, expected)

    def test_anchor_on_invalid_not_subtotal(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.invalid_subtotal_2, dim)
        expected = None
        actual = subtotal.anchor
        self.assertEqual(actual, expected)

    @patch('cr.cube.subtotal.Subtotal._all_dim_ids', [1, 3, 5])
    def test_anchor_on_valid(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.valid_subtotal, dim)
        expected = 5
        actual = subtotal.anchor
        self.assertEqual(actual, expected)

    def test_args_on_invalid_1(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.invalid_subtotal_1, dim)
        expected = []
        actual = subtotal.args
        self.assertEqual(actual, expected)

    def test_args_on_invalid_2(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.invalid_subtotal_2, dim)
        expected = []
        actual = subtotal.args
        self.assertEqual(actual, expected)

    def test_args_on_valid(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.valid_subtotal, dim)
        expected = [5, 4]
        actual = subtotal.args
        self.assertEqual(actual, expected)

    def test_anchor_on_uppercased_bottom(self):
        dim = Mock()
        dim.elements.return_value = [{'id': 5}, {'id': 4}]
        subtotal = Subtotal(self.valid_subtotal_anchor_bottom, dim)
        anchor = subtotal.anchor
        assert anchor == 'bottom'

