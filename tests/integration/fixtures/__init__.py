'''Home of the fixtures for Crunch Cube integration tests.

The tests are considered 'integration', since they need to load the fixtures
from the files, however they're very close to unit tests. The reason for using
the fixtures is better visibility of the query data, and also keeping the
source files relatively clean.
'''
import os
import json


def load_fixture(filename):
    '''Loads fixtures for CrunchCube integration tests.'''
    cubes_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'cubes')
    with open(os.path.join(cubes_directory, filename)) as ctx_file:
        fixture = json.load(ctx_file)
    return fixture


# Bivariate Cubes
FIXT_CAT_X_CAT = load_fixture('cat-x-cat.json')
FIXT_CAT_X_DATETIME = load_fixture('cat-x-datetime.json')

# Univariate Cubes
FIXT_UNIVARIATE_CATEGORICAL = load_fixture('univariate-categorical.json')
FIXT_VOTER_REGISTRATION = load_fixture('voter-registration.json')
FIXT_SIMPLE_DATETIME = load_fixture('simple-datetime.json')
FIXT_SIMPLE_TEXT = load_fixture('simple-text.json')
FIXT_SIMPLE_CAT_ARRAY = load_fixture('simple-cat-array.json')

# Various other Cubes
FIXT_CAT_X_NUM_X_DATETIME = load_fixture('cat-x-num-x-datetime.json')
FIXT_SIMPLE_MR = load_fixture('simple-mr.json')
FIXT_CAT_X_MR = load_fixture('cat-x-mr.json')
FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED = load_fixture(
    'econ-gender-x-ideology-weighted.json')
FIXT_CAT_X_CAT_GERMAN_WEIGHTED = load_fixture('cat-x-cat-german-weighted.json')
FIXT_STATS_TEST = load_fixture('stats_test.json')
