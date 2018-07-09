import os
from functools import partial

from cr.cube.utils import load_fixture

CUBES_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(cube_file):
    load = partial(load_fixture, CUBES_DIR)
    return load(cube_file)


CA_CAT_X_ITEMS = _load('ca-cat-x-items.json')
CA_ITEMS_X_CAT = _load('ca-items-x-cat.json')
CA_X_MR = _load('ca-x-mr.json')
CAT_X_CA_CAT_X_ITEMS = _load('cat-x-ca-cat-x-items.json')
CAT_X_CAT = _load('cat-x-cat.json')
CAT_X_MR = _load('cat-x-mr.json')
MR_X_CAT = _load('mr-x-cat.json')
UNIVARIATE_CAT = _load('univariate-cat.json')
