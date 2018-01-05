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
FIXT_CAT_X_MR_SIMPLE = load_fixture(CUBES_DIR, 'cat-x-mr.json')
FIXT_ECON_GENDER_X_IDEOLOGY_WEIGHTED = load_fixture(
    CUBES_DIR,
    'econ-gender-x-ideology-weighted.json',
)
FIXT_CAT_X_CAT_GERMAN_WEIGHTED = load_fixture(CUBES_DIR,
                                              'cat-x-cat-german-weighted.json')
FIXT_STATS_TEST = load_fixture(CUBES_DIR, 'stats_test.json')
FIXT_ECON_MEAN_AGE_BLAME_X_GENDER = load_fixture(
    CUBES_DIR, 'econ-mean-age-blame-x-gender.json',
)
FIXT_ECON_MEAN_NO_DIMS = load_fixture(CUBES_DIR, 'econ-mean-no-dims.json')
FIXT_MR_X_CAT_PROFILES_STATS_WEIGHTED = load_fixture(
    CUBES_DIR,
    'mr-x-cat-profiles-stats-weighted.json',
)
FIXT_ADMIT_X_DEPT_UNWEIGHTED = load_fixture(CUBES_DIR,
                                            'admit-x-dept-unweighted.json')
FIXT_ADMIT_X_GENDER_WEIGHTED = load_fixture(CUBES_DIR,
                                            'admit-x-gender-weighted.json')
FIXT_SELECTED_CROSSTAB_4 = load_fixture(CUBES_DIR, 'selected-crosstab-4.json')
FIXT_PETS_X_PETS = load_fixture(CUBES_DIR, 'pets-x-pets.json')
FIXT_PETS_X_FRUIT = load_fixture(CUBES_DIR, 'pets-x-fruit.json')
FIXT_PETS_ARRAY = load_fixture(CUBES_DIR, 'pets-array.json')
FIXT_ECON_BLAME_WITH_HS = load_fixture(CUBES_DIR, 'econ-blame-with-hs.json')
FIXT_ECON_BLAME_WITH_HS_MISSING = load_fixture(
    CUBES_DIR,
    'econ-blame-with-hs-missing.json',
)
FIXT_ECON_BLAME_X_IDEOLOGY_ROW_HS = load_fixture(
    CUBES_DIR,
    'econ-blame-x-ideology-row-hs.json',
)
FIXT_ECON_BLAME_X_IDEOLOGY_COL_HS = load_fixture(
    CUBES_DIR,
    'econ-blame-x-ideology-col-hs.json',
)
FIXT_ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS = load_fixture(
    CUBES_DIR,
    'econ-blame-x-ideology-row-and-col-hs.json',
)
FIXT_SIMPLE_CA_HS = load_fixture(CUBES_DIR, 'simple-ca-hs.json')
FIXT_FRUIT_X_PETS = load_fixture(CUBES_DIR, 'fruit-x-pets.json')
FIXT_ECON_US_PROBLEM_X_BIGGER_PROBLEM = load_fixture(
    CUBES_DIR,
    'econ-us-problem-x-bigger-problem.json',
)
FIXT_PETS_ARRAY_X_PETS = load_fixture(
    CUBES_DIR,
    'pets-array-x-pets.json',
)
FIXT_PETS_X_PETS_ARRAY = load_fixture(
    CUBES_DIR,
    'pets-x-pets-array.json',
)
FIXT_FRUIT_X_PETS_ARRAY = load_fixture(
    CUBES_DIR,
    'fruit-x-pets-array.json'
)
FIXT_IDENTITY_X_PERIOD = load_fixture(
    CUBES_DIR,
    'econ-identity-x-period.json'
)
FIXT_CA_SINGLE_CAT = load_fixture(CUBES_DIR, 'cat-arr-with-single-cat.json')
FIXT_MR_X_SINGLE_WAVE = load_fixture(CUBES_DIR, 'mr-x-single-wave.json')
FIXT_SELECTED_3_WAY_2 = load_fixture(
    CUBES_DIR,
    'selected-3way-2-filledmissing.json'
)
FIXT_SELECTED_3_WAY = load_fixture(
    CUBES_DIR,
    'selected-3way-filledmissing.json'
)
FIXT_ARRAY_X_MR = load_fixture(CUBES_DIR, 'array-by-mr.json')
FIXT_PROFILES_PERCENTS = load_fixture(CUBES_DIR,
                                      'test-profiles-percentages.json')
FIXT_CAT_X_CAT_WITH_EMPTY_COLS = load_fixture(CUBES_DIR,
                                              'cat-x-cat-with-empty-cols.json')
FIXT_BINNED = load_fixture(CUBES_DIR, 'binned.json')
FIXT_SINGLE_COL_MARGIN_NOT_ITERABLE = load_fixture(
    CUBES_DIR,
    'single-col-margin-not-iterable.json'
)
FIXT_GENDER_PARTY_RACE = load_fixture(CUBES_DIR, 'gender-party-race.json')
FIXT_SEL_ARR_FIRST = load_fixture(CUBES_DIR,
                                  'selected-crosstab-array-first.json')
FIXT_SEL_ARR_LAST = load_fixture(CUBES_DIR, 'selected-crosstab-array-last.json')
FIXT_MR_X_MR = load_fixture(CUBES_DIR, 'selected-by-selected.json')
