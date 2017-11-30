'''Home of the fixtures for Crunch Cube integration tests.

The tests are considered 'integration', since they need to load the fixtures
from the files, however they're very close to unit tests. The reason for using
the fixtures is better visibility of the query data, and also keeping the
source files relatively clean.
'''

import os
from cr.cube.utils import load_fixture

CUBES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cubes')


# Bivariate Cubes
FIXT_CAT_X_CAT = load_fixture(CUBES_DIR, 'cat-x-cat.json')
FIXT_CAT_X_DATETIME = load_fixture(CUBES_DIR, 'cat-x-datetime.json')

# Univariate Cubes
FIXT_UNIVARIATE_CATEGORICAL = load_fixture(CUBES_DIR,
                                           'univariate-categorical.json')
FIXT_VOTER_REGISTRATION = load_fixture(CUBES_DIR, 'voter-registration.json')
FIXT_SIMPLE_DATETIME = load_fixture(CUBES_DIR, 'simple-datetime.json')
FIXT_SIMPLE_TEXT = load_fixture(CUBES_DIR, 'simple-text.json')
FIXT_SIMPLE_CAT_ARRAY = load_fixture(CUBES_DIR, 'simple-cat-array.json')

# Various other Cubes
FIXT_CAT_X_NUM_X_DATETIME = load_fixture(CUBES_DIR,
                                         'cat-x-num-x-datetime.json')
FIXT_SIMPLE_MR = load_fixture(CUBES_DIR, 'simple-mr.json')
FIXT_CAT_X_MR = load_fixture(CUBES_DIR, 'cat-x-mr.json')
FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED = load_fixture(
    CUBES_DIR,
    'econ-gender-x-ideology-weighted.json'
)
FIXT_CAT_X_CAT_GERMAN_WEIGHTED = load_fixture(CUBES_DIR,
                                              'cat-x-cat-german-weighted.json')
FIXT_STATS_TEST = load_fixture(CUBES_DIR, 'stats_test.json')
FIXT_ECON_MEAN_AGE_BLAME_X_GENDER = load_fixture(
    CUBES_DIR, 'econ-mean-age-blame-x-gender.json'
)
FIXT_ECON_MEAN_NO_DIMS = load_fixture(CUBES_DIR, 'econ-mean-no-dims.json')
FIXT_MR_X_CAT_PROFILES_STATS_WEIGHTED = load_fixture(
    CUBES_DIR,
    'mr-x-cat-profiles-stats-weighted.json'
)
FIXT_ADMIT_X_DEPT_UNWEIGHTED = load_fixture(CUBES_DIR,
                                            'admit-x-dept-unweighted.json')
FIXT_ADMIT_X_GENDER_WEIGHTED = load_fixture(CUBES_DIR,
                                            'admit-x-gender-weighted.json')
FIXT_SELECTED_CROSSTAB_4 = load_fixture(CUBES_DIR, 'selected-crosstab-4.json')
FIXT_PETS_X_PETS = load_fixture(CUBES_DIR, 'pets-x-pets.json')
FIXT_PETS_X_FRUIT = load_fixture(CUBES_DIR, 'pets-x-fruit.json')
FIXT_PETS_ARRAY = load_fixture(CUBES_DIR, 'pets-array.json')
