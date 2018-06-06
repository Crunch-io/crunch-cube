'''Utility functions for crunch cube, as well as other modules.'''
import os
import json


def load_fixture(fixtures_directory, filename):
    '''Loads fixtures for CrunchCube integration tests.'''
    with open(os.path.join(fixtures_directory, filename)) as ctx_file:
        fixture = json.load(ctx_file)
    return fixture


def lazyproperty(func):
    """@lazyproperty decorator

    Decorated method will be called only on first access to calculate
    a cached property value. After that, the cached value is returned. Note
    that the cached value is stored in a variable with the same name as the
    decorated property, with the prefix '_cached_' prepended, such that the
    cached value of property `foobar` is stored in `_cached_foobar`. Be aware
    to avoid naming collisions.
    """
    cache_attr_name = '_cached_%s' % func.__name__
    docstring = func.__doc__

    def get_prop_value(obj):
        try:
            return getattr(obj, cache_attr_name)
        except AttributeError:
            value = func(obj)
            setattr(obj, cache_attr_name, value)
            return value

    return property(get_prop_value, doc=docstring)
