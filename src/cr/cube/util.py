# encoding: utf-8

"""Utility functions for crunch cube, as well as other modules."""

import collections
import functools

import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # pragma: no cover
    from itertools import filterfalse as ifilterfalse


def calculate_overlap_tstats(
    cls, mr_dimensions, mr_counts, mr_base_counts, mr_counts_with_missings
):
    numerator = np.zeros(np.array(mr_counts.shape)[[0, 1, 3]])
    standard_error = np.zeros(np.array(mr_counts.shape)[[0, 1, 3]])
    for slice_index in range(mr_counts.shape[0]):
        overlap_slice = cls(
            mr_dimensions,
            mr_counts[slice_index],
            mr_base_counts[slice_index],
            mr_counts_with_missings[slice_index],
        )
        diff, se_diff = overlap_slice.tstats_overlap
        numerator[slice_index] = diff
        standard_error[slice_index] = se_diff
    return numerator, standard_error


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


def intersperse_hs_in_std_res(slice_, hs_dims, res):
    """Perform the insertions of place-holding rows and cols for insertions."""
    for dim, inds in enumerate(slice_.inserted_hs_indices()):
        if dim not in hs_dims:
            continue
        for i in inds:
            res = np.insert(res, i, np.nan, axis=(dim - slice_.ndim))
    return res


class Counter(dict):
    """Mapping where default values are zero"""

    def __missing__(self, key):
        return 0


class lazyproperty(object):
    """Decorator like @property, but evaluated only on first access.

    Like @property, this can only be used to decorate methods having only
    a `self` parameter, and is accessed like an attribute on an instance,
    i.e. trailing parentheses are not used. Unlike @property, the decorated
    method is only evaluated on first access; the resulting value is cached
    and that same value returned on second and later access without
    re-evaluation of the method.

    Like @property, this class produces a *data descriptor* object, which is
    stored in the __dict__ of the *class* under the name of the decorated
    method ('fget' nominally). The cached value is stored in the __dict__ of
    the *instance* under that same name.

    Because it is a data descriptor (as opposed to a *non-data descriptor*),
    its `__get__()` method is executed on each access of the decorated
    attribute; the __dict__ item of the same name is "shadowed" by the
    descriptor.

    While this may represent a performance improvement over a property, its
    greater benefit may be its other characteristics. One common use is to
    construct collaborator objects, removing that "real work" from the
    constructor, while still only executing once. It also de-couples client
    code from any sequencing considerations; if it's accessed from more than
    one location, it's assured it will be ready whenever needed.

    Loosely based on: https://stackoverflow.com/a/6849299/1902513.

    A lazyproperty is read-only. There is no counterpart to the optional
    "setter" (or deleter) behavior of an @property. This is critically
    important to maintaining its immutability and idempotence guarantees.
    Attempting to assign to a lazyproperty raises AttributeError
    unconditionally.

    The parameter names in the methods below correspond to this usage
    example::

        class Obj(object)

            @lazyproperty
            def fget(self):
                return 'some result'

        obj = Obj()

    Not suitable for wrapping a function (as opposed to a method) because it
    is not callable.
    """

    def __init__(self, fget):
        """*fget* is the decorated method (a "getter" function).

        A lazyproperty is read-only, so there is only an *fget* function (a
        regular @property can also have an fset and fdel function). This name
        was chosen for consistency with Python's `property` class which uses
        this name for the corresponding parameter.
        """
        # ---maintain a reference to the wrapped getter method
        self._fget = fget
        # ---adopt fget's __name__, __doc__, and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, type=None):
        """Called on each access of 'fget' attribute on class or instance.

        *self* is this instance of a lazyproperty descriptor "wrapping" the
        property method it decorates (`fget`, nominally).

        *obj* is the "host" object instance when the attribute is accessed
        from an object instance, e.g. `obj = Obj(); obj.fget`. *obj* is None
        when accessed on the class, e.g. `Obj.fget`.

        *type* is the class hosting the decorated getter method (`fget`) on
        both class and instance attribute access.
        """
        # ---when accessed on class, e.g. Obj.fget, just return this
        # ---descriptor instance (patched above to look like fget).
        if obj is None:
            return self

        # ---when accessed on instance, start by checking instance __dict__
        value = obj.__dict__.get(self.__name__)
        if value is None:
            # ---on first access, __dict__ item will absent. Evaluate fget()
            # ---and store that value in the (otherwise unused) host-object
            # ---__dict__ value of same name ('fget' nominally)
            value = self._fget(obj)
            obj.__dict__[self.__name__] = value
        return value

    def __set__(self, obj, value):
        """Raises unconditionally, to preserve read-only behavior.

        This decorator is intended to implement immutable (and idempotent)
        object attributes. For that reason, assignment to this property must
        be explicitly prevented.

        If this __set__ method was not present, this descriptor would become
        a *non-data descriptor*. That would be nice because the cached value
        would be accessed directly once set (__dict__ attrs have precedence
        over non-data descriptors on instance attribute lookup). The problem
        is, there would be nothing to stop assignment to the cached value,
        which would overwrite the result of `fget()` and break both the
        immutability and idempotence guarantees of this decorator.

        The performance with this __set__() method in place was roughly 0.4
        usec per access when measured on a 2.8GHz development machine; so
        quite snappy and probably not a rich target for optimization efforts.
        """
        raise AttributeError("can't set attribute")


def lru_cache(maxsize=100):  # pragma: no cover
    """Least-recently-used cache decorator.

    Arguments to the cached function must be hashable.
    Cache performance statistics stored in f.hits and f.misses.
    Clear the cache with f.clear().
    http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used
    """
    maxqueue = maxsize * 10

    def decorating_function(
        user_function, len=len, iter=iter, tuple=tuple, sorted=sorted, KeyError=KeyError
    ):
        cache = {}  # mapping of args to results
        queue = collections.deque()  # order that keys have been used
        refcount = Counter()  # times each key is in the queue
        sentinel = object()  # marker for looping around the queue
        kwd_mark = object()  # separate positional and keyword args

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
                for key in ifilterfalse(
                    refcount.__contains__, iter(queue_pop, sentinel)
                ):
                    queue_appendleft(key)
                    refcount[key] = 1

            return result

        def clear():  # pragma: no cover
            cache.clear()
            queue.clear()
            refcount.clear()
            wrapper.hits = wrapper.misses = 0

        wrapper.hits = wrapper.misses = 0
        wrapper.clear = clear
        return wrapper

    return decorating_function


memoize = lru_cache(100)
