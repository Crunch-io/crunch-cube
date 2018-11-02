# encoding: utf-8

"""Unit test suite for cr.cube.util module."""

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np
import pytest

from cr.cube.util import compress_pruned, lazyproperty


class Describe_compress_pruned(object):

    def it_returns_an_unmasked_array_unchanged(self):
        table = np.zeros((2, 2))
        return_value = compress_pruned(table)
        assert return_value is table

    def it_returns_the_data_of_a_0D_masked_array(self):
        table = np.ma.masked_array(np.array(42), True)
        return_value = compress_pruned(table)
        assert return_value == 42

    def it_returns_a_compressed_1D_masked_array(self):
        table = np.ma.masked_array(
            np.array([42, 43, 44]), [False, True, False]
        )
        return_value = compress_pruned(table)
        np.testing.assert_array_equal(return_value, [42, 44])

    def it_NaNs_isolated_pruned_float_values(self):
        table = np.ma.masked_array(
            np.array([[4.2, 0.0, 4.4],
                      [4.5, 4.6, 4.7]]),
            [[False, True, False],
             [False, False, False]],
            dtype=float
        )
        return_value = compress_pruned(table)
        np.testing.assert_array_equal(
            return_value,
            [[4.2, np.nan, 4.4],
             [4.5, 4.6, 4.7]]
        )


class DescribeLazyPropertyDecorator(object):
    """Tests @lazyproperty decorator class."""

    def it_is_a_lazyproperty_object_on_class_access(self, Obj):
        assert isinstance(Obj.fget, lazyproperty)

    def but_it_adopts_the_name_of_the_decorated_method(self, Obj):
        assert Obj.fget.__name__ == 'fget'

    def and_it_adopts_the_module_of_the_decorated_method(self, Obj):
        # ---the module name actually, not a module object
        assert Obj.fget.__module__ == __name__

    def and_it_adopts_the_docstring_of_the_decorated_method(self, Obj):
        assert Obj.fget.__doc__ == 'Docstring of Obj.fget method definition.'

    def it_only_calculates_value_on_first_call(self, obj):
        assert obj.fget == 1
        assert obj.fget == 1

    def it_raises_on_attempt_to_assign(self, obj):
        assert obj.fget == 1
        with pytest.raises(AttributeError):
            obj.fget = 42
        assert obj.fget == 1
        assert obj.fget == 1

    # fixture components ---------------------------------------------

    @pytest.fixture
    def Obj(self):
        class Obj(object):
            @lazyproperty
            def fget(self):
                """Docstring of Obj.fget method definition."""
                if not hasattr(self, '_n'):
                    self._n = 0
                self._n += 1
                return self._n
        return Obj

    @pytest.fixture
    def obj(self, Obj):
        return Obj()
