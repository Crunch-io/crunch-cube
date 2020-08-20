# encoding: utf-8

"""Partial-integration test suite for `cr.cube.matrix` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

from cr.cube.dimension import _Subtotal
from cr.cube.matrix import _BaseInsertedVector


class Describe_BaseInsertedVector(object):
    """Unit test suite for `cr.cube.matrix._BaseInsertedVector` object."""

    def it_knows_its_insertion_id_when_it_has_been_assigned_one(self):
        subtotal_dict = {"insertion_id": 42}
        subtotal = _Subtotal(subtotal_dict, None, None, None)
        inserted_vector = _BaseInsertedVector(subtotal, None, None, None, None)

        insertion_id = inserted_vector.insertion_id

        assert insertion_id == 42

    def and_it_is_assigned_a_fallback_when_it_has_no_insertion_id(self):
        subtotal_dict = {}
        subtotal = _Subtotal(subtotal_dict, None, None, 24)
        inserted_vector = _BaseInsertedVector(subtotal, None, None, None, None)

        insertion_id = inserted_vector.insertion_id

        assert insertion_id == 24
