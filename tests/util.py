# encoding: utf-8

"""Utilities for cr.cube tests."""

from __future__ import absolute_import, division, print_function, unicode_literals

import os


def load_expectation(expectation_file_name, strip=True):  # pragma: no cover
    """Return (unicode) str containing text in *expectation_file_name*.

    Expectation file path is rooted at tests/expectations.
    """
    thisdir = os.path.dirname(__file__)
    expectation_file_path = os.path.abspath(
        os.path.join(thisdir, "expectations", "%s.txt" % expectation_file_name)
    )
    with open(expectation_file_path, "rb") as f:
        expectation_bytes = f.read()
    if strip:
        return expectation_bytes.decode("utf-8").strip()
    return expectation_bytes.decode("utf-8")
