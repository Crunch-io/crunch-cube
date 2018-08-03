'''Utility functions for crunch cube, as well as other modules.'''
import collections
import functools
from itertools import ifilterfalse
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


def lru_cache(maxsize=100):
    '''Least-recently-used cache decorator.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    Clear the cache with f.clear().
    http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used

    '''
    maxqueue = maxsize * 10

    def decorating_function(user_function,
            len=len, iter=iter, tuple=tuple, sorted=sorted, KeyError=KeyError):
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
