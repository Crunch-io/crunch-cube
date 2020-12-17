# encoding: utf-8

"""JSON cube-response source files for testing purposes."""

import json
import os

CUBES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cubes")


class LazyCubeResponseLoader(object):
    """Loads and caches cube-responses by name from fixture directory.

    Provides access to all the cube-response fixtures in a directory by
    a standardized mapping of the file name, e.g. cat-x-cat.json is available
    as the `.CAT_X_CAT` attribute of the loader.

    The fixture directory is specified relative to this (fixture root)
    directory.
    """

    def __init__(self, relpath):
        self._relpath = relpath
        self._cache = {}

    def __getattr__(self, fixture_name):
        """Return cube-dict from JSON file mapping to *fixture_name*.

        A *fixture_name* like 'CAT_X_CAT' will map to the JSON file
        'cat-x-cat.json' in the directory specified on construction.
        """
        if fixture_name not in self._cache:
            self._load_to_cache(fixture_name)
        return self._cache[fixture_name]

    @property
    def _dirpath(self):
        """Absolute path to relative directory specified in relpath."""
        thisdir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(thisdir, self._relpath))

    def _json_path(self, fixture_name):
        """Return absolute path to JSON file for *fixture_name*."""
        return "%s/%s.json" % (self._dirpath, fixture_name.replace("_", "-").lower())

    def _load_json(self, path):
        """Return dict parsed from JSON at *path*."""
        with open(path) as f:
            cube_response = json.load(f)
        return cube_response

    def _load_to_cache(self, fixture_name):
        json_path = self._json_path(fixture_name)
        if not os.path.exists(json_path):
            raise ValueError("no JSON fixture found at %s" % json_path)
        self._cache[fixture_name] = self._load_json(json_path)


CR = LazyCubeResponseLoader(".")  # ---mnemonic: CR = 'cube-response'---
NA = LazyCubeResponseLoader("./numeric_arrays")
SM = LazyCubeResponseLoader("./scale_means")
TR = LazyCubeResponseLoader("./transforms")
