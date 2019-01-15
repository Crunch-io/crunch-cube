# encoding: utf-8

"""Functions that make mocking with pytest easier and more readable."""

from __future__ import absolute_import, division, print_function, unicode_literals

from mock import ANY, call, Mock  # noqa
from mock import create_autospec, patch, PropertyMock


def class_mock(request, q_class_name, autospec=True, **kwargs):
    """Return mock patching class with qualified name *q_class_name*.

    The mock is autospec'ed based on the patched class unless the optional
    argument *autospec* is set to False. Any other keyword arguments are
    passed through to Mock(). Patch is reversed after calling test returns.
    """
    _patch = patch(q_class_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def function_mock(
    request, q_function_name, autospec=True, **kwargs
):  # pragma: no cover
    """Return mock patching function with qualified name *q_function_name*.

    Patch is reversed after calling test returns.
    """
    _patch = patch(q_function_name, autospec=autospec, **kwargs)
    request.addfinalizer(_patch.stop)
    return _patch.start()


def initializer_mock(request, cls, autospec=True, **kwargs):
    """Return mock for __init__() method on *cls*.

    The patch is reversed after pytest uses it.
    """
    _patch = patch.object(
        cls, "__init__", autospec=autospec, return_value=None, **kwargs
    )
    request.addfinalizer(_patch.stop)
    return _patch.start()


def instance_mock(request, cls, name=None, spec_set=True, **kwargs):
    """Return mock for instance of *cls* that draws its spec from the class.

    The mock will not allow new attributes to be set on the instance. If
    *name* is missing or |None|, the name of the returned |Mock| instance is
    set to *request.fixturename*. Additional keyword arguments are passed
    through to the Mock() call that creates the mock.
    """
    name = name if name is not None else request.fixturename
    return create_autospec(cls, _name=name, spec_set=spec_set, instance=True, **kwargs)


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
