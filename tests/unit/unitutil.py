# encoding: utf-8

"""Functions that make mocking with pytest easier and more readable."""

from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from mock import ANY, call  # noqa
from mock import create_autospec, patch, PropertyMock


def instance_mock(request, cls, name=None, spec_set=True, **kwargs):
    """Return mock for instance of *cls* that draws its spec from the class.

    The mock will not allow new attributes to be set on the instance. If
    *name* is missing or |None|, the name of the returned |Mock| instance is
    set to *request.fixturename*. Additional keyword arguments are passed
    through to the Mock() call that creates the mock.
    """
    name = name if name is not None else request.fixturename
    return create_autospec(
        cls, _name=name, spec_set=spec_set, instance=True, **kwargs
    )


def method_mock(request, cls, method_name, autospec=True, **kwargs):
    """Return mock for method *method_name* on *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(cls, method_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def property_mock(request, cls, prop_name, **kwargs):
    """Return mock for property *prop_name* on class *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(cls, prop_name, new_callable=PropertyMock, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()
