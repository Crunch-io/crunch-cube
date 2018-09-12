'''Utility functions for crunch cube, as well as other modules.'''
import collections
import functools
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

import json
import os


class Counter(dict):
    """Mapping where default values are zero"""
    def __missing__(self, key):
        return 0


def load_fixture(fixtures_directory, filename):
    """Loads fixtures for CrunchCube integration tests."""
    with open(os.path.join(fixtures_directory, filename)) as ctx_file:
        fixture = json.load(ctx_file)
    return fixture


class lazyproperty(property):
    """borrowed from: https://stackoverflow.com/questions/3012421/python-memoising-deferred-lookup-property-decorator"""  # noqa
    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, None)
        if value is None:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


def lru_cache(maxsize=100):
    '''Least-recently-used cache decorator.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    Clear the cache with f.clear().
    http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    '''
    maxqueue = maxsize * 10

    def decorating_function(user_function, len=len, iter=iter, tuple=tuple,
                            sorted=sorted, KeyError=KeyError):
        cache = {}                   # mapping of args to results
        queue = collections.deque()  # order that keys have been used
        refcount = Counter()         # times each key is in the queue
        sentinel = object()          # marker for looping around the queue
        kwd_mark = object()          # separate positional and keyword args

        # lookup optimizations (ugly but fast)
        queue_append, queue_popleft = queue.append, queue.popleft
        queue_appendleft, queue_pop = queue.appendleft, queue.pop

        @functools.wraps(user_function)
        def wrapper(*args, **kwds):
            # cache key records both positional and keyword args
            key = args
            if kwds:
                key += (kwd_mark,) + tuple(sorted(kwds.items()))

            # record recent use of this key
            queue_append(key)
            refcount[key] += 1

            # get cache entry or compute if not found
            try:
                result = cache[key]
                wrapper.hits += 1
            except KeyError:
                result = user_function(*args, **kwds)
                cache[key] = result
                wrapper.misses += 1

                # purge least recently used cache entry
                if len(cache) > maxsize:
                    key = queue_popleft()
                    refcount[key] -= 1
                    while refcount[key]:
                        key = queue_popleft()
                        refcount[key] -= 1
                    del cache[key], refcount[key]

            # periodically compact the queue by eliminating duplicate keys
            # while preserving order of most recent access
            if len(queue) > maxqueue:
                refcount.clear()
                queue_appendleft(sentinel)
                for key in ifilterfalse(refcount.__contains__,
                                        iter(queue_pop, sentinel)):
                    queue_appendleft(key)
                    refcount[key] = 1

            return result

        def clear():
            cache.clear()
            queue.clear()
            refcount.clear()
            wrapper.hits = wrapper.misses = 0

        wrapper.hits = wrapper.misses = 0
        wrapper.clear = clear
        return wrapper
    return decorating_function


memoize = lru_cache(100)


def compress_pruned(table):
    """Compress table based on pruning mask.

    Only the rows/cols in which all of the elements are masked need to be
    pruned.
    """
    if not isinstance(table, np.ma.core.MaskedArray):
        return table

    if table.ndim == 0:
        return table.data

    if table.ndim == 1:
        return np.ma.compressed(table)

    row_inds = ~table.mask.all(axis=1)
    col_inds = ~table.mask.all(axis=0)
    table = table[row_inds, :][:, col_inds]
    if table.dtype == float and table.mask.any():
        table[table.mask] = np.nan
    return table
