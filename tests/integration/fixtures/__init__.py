'''Home of the fixtures for Crunch Cube integration tests.

The tests are considered 'integration', since they need to load the fixtures
from the files, however they're very close to unit tests. The reason for using
the fixtures is better visibility of the query data, and also keeping the
source files relatively clean.
'''

import os
from functools import partial

from cr.cube.utils import load_fixture

CUBES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cubes')


def _load(cube_file):
    load = partial(load_fixture, CUBES_DIR)
    return load(cube_file)


# Bivariate Cubes
CAT_X_CAT = _load('cat-x-cat.json')
CAT_X_DATETIME = _load('cat-x-datetime.json')
LOGICAL_X_CAT = _load('logical-x-cat.json')

# Univariate Cubes
UNIVARIATE_CATEGORICAL = _load('univariate-categorical.json')
VOTER_REGISTRATION = _load('voter-registration.json')
SIMPLE_DATETIME = _load('simple-datetime.json')
SIMPLE_TEXT = _load('simple-text.json')
SIMPLE_CAT_ARRAY = _load('simple-cat-array.json')
LOGICAL_UNIVARIATE = _load('logical-univariate.json')

# Various other Cubes
CAT_X_NUM_X_DATETIME = _load('cat-x-num-x-datetime.json')
SIMPLE_MR = _load('simple-mr.json')
CAT_X_MR_SIMPLE = _load('cat-x-mr.json')
CAT_X_MR_PRUNED_ROW = _load('cat-x-mr-pruned-row.json')
CAT_X_MR_PRUNED_COL = _load('cat-x-mr-pruned-col.json')
CAT_X_MR_PRUNED_ROW_COL = _load('cat-x-mr-pruned-row-col.json')
MR_X_CAT_PRUNED_COL = _load('mr-x-cat-pruned-col.json')
MR_X_CAT_PRUNED_ROW = _load('mr-x-cat-pruned-row.json')
MR_X_CAT_PRUNED_ROW_COL = _load('mr-x-cat-pruned-row-col.json')
ECON_GENDER_X_IDEOLOGY_WEIGHTED = _load('econ-gender-x-ideology-weighted.json')
CAT_X_CAT_GERMAN_WEIGHTED = _load('cat-x-cat-german-weighted.json')
STATS_TEST = _load('stats_test.json')
ECON_MEAN_AGE_BLAME_X_GENDER = _load(
    'econ-mean-age-blame-x-gender.json',
)
ECON_MEAN_NO_DIMS = _load('econ-mean-no-dims.json')
MR_X_CAT_PROFILES_STATS_WEIGHTED = _load(
    'mr-x-cat-profiles-stats-weighted.json',
)
ADMIT_X_DEPT_UNWEIGHTED = _load('admit-x-dept-unweighted.json')
ADMIT_X_GENDER_WEIGHTED = _load('admit-x-gender-weighted.json')
SELECTED_CROSSTAB_4 = _load('selected-crosstab-4.json')
PETS_X_PETS = _load('pets-x-pets.json')
PETS_X_FRUIT = _load('pets-x-fruit.json')
PETS_ARRAY = _load('pets-array.json')
PETS_ARRAY_CAT_FIRST = _load('pets-array-cat-first.json')
PETS_ARRAY_SUBVAR_FIRST = _load('pets-array-subvar-first.json')
ECON_BLAME_WITH_HS = _load('econ-blame-with-hs.json')
ECON_BLAME_WITH_HS_MISSING = _load('econ-blame-with-hs-missing.json')
ECON_BLAME_X_IDEOLOGY_ROW_HS = _load('econ-blame-x-ideology-row-hs.json')
ECON_BLAME_X_IDEOLOGY_COL_HS = _load('econ-blame-x-ideology-col-hs.json')
ECON_BLAME_X_IDEOLOGY_ROW_AND_COL_HS = _load(
    'econ-blame-x-ideology-row-and-col-hs.json'
)
SIMPLE_CA_HS = _load('simple-ca-hs.json')
FRUIT_X_PETS = _load('fruit-x-pets.json')
ECON_US_PROBLEM_X_BIGGER_PROBLEM = _load(
    'econ-us-problem-x-bigger-problem.json'
)
PETS_ARRAY_X_PETS = _load('pets-array-x-pets.json')
PETS_X_PETS_ARRAY = _load('pets-x-pets-array.json')
FRUIT_X_PETS_ARRAY = _load('fruit-x-pets-array.json')
IDENTITY_X_PERIOD = _load('econ-identity-x-period.json')
CA_SINGLE_CAT = _load('cat-arr-with-single-cat.json')
MR_X_SINGLE_WAVE = _load('mr-x-single-wave.json')
SELECTED_3_WAY_2 = _load('selected-3way-2-filledmissing.json')
SELECTED_3_WAY = _load('selected-3way-filledmissing.json')
ARRAY_X_MR = _load('array-by-mr.json')
PROFILES_PERCENTS = _load('test-profiles-percentages.json')
CAT_X_CAT_WITH_EMPTY_COLS = _load('cat-x-cat-with-empty-cols.json')
BINNED = _load('binned.json')
SINGLE_COL_MARGIN_NOT_ITERABLE = _load(
    'single-col-margin-not-iterable.json'
)
GENDER_PARTY_RACE = _load('gender-party-race.json')
CAT_X_ITEMS_X_CATS_HS = _load('cat-x-items-x-cats-hs.json')
SEL_ARR_FIRST = _load('selected-crosstab-array-first.json')
SEL_ARR_LAST = _load('selected-crosstab-array-last.json')
MR_X_MR = _load('selected-by-selected.json')
MR_X_MR_HETEROGENOUS = _load('mr-by-mr-different-mrs.json')
SINGLE_CAT_MEANS = _load('means-with-single-cat.json')
FRUIT_HS_TOP_BOTTOM = _load('fruit-hs-top-bottom.json')
FRUIT_X_PETS_HS_TOP_BOTTOM = _load('fruit-x-pets-hs-top-bottom.json')
CA_X_SINGLE_CAT = _load('ca-x-single-cat.json')
CA_WITH_NETS = _load('ca-with-nets.json')
CAT_X_DATE_HS_PRUNE = _load('cat-x-date-hs-prune.json')
CAT_X_NUM_HS_PRUNE = _load('cat-x-num-hs-prune.json')
CA_SUBVAR_X_CAT_HS = _load('ca-subvar-x-cat-hs.json')
CAT_X_MR_X_MR = _load('cat-x-mr-x-mr.json')
CAT_X_MR_X_MR_PRUNED_ROWS = _load('cat-x-mr-x-mr-pruned-rows.json')
CAT_X_MR_WEIGHTED_HS = _load('cat-x-mr-weighted-hs.json')
FRUIT_X_PETS_ARRAY_SUBVARS_FIRST = _load(
    'fruit-x-pets-array-subvars-first.json'
)
FRUIT_X_PETS_ARRAY_PETS_FIRST = _load(
    'fruit-x-pets-array-pets-first.json'
)
PROMPTED_AWARENESS = _load('prompted-awareness.json')
SCALE_WITH_NULL_VALUES = _load('scale-with-null-values.json')
PETS_X_FRUIT_HS = _load('pets-x-fruit-hs.json')
VALUE_SERVICES = _load('value-added-services.json')
LETTERS_X_PETS_HS = _load('letters-x-pets-hs.json')
MISSING_CAT_HS = _load('missing-cat-hs.json')
XYZ_SIMPLE_ALLTYPES = _load('xyz-simple-alltypes.json')
MR_X_MR_X_CAT = _load('mr-mr-cat.json')
MR_X_CAT_X_MR = _load('mr-cat-mr.json')
CA_X_CAT_HS = _load('ca-x-cat-hs.json')
BBC_NEWS = _load('bbc-news.json')
AGE_X_ACCRPIPE = _load('age-x-accrpipe.json')
CAT_X_MR_X_CAT_MISSING = _load('cat-x-mr-x-cat-missing.json')
MR_X_CA_HS = _load('mr-x-ca-with-hs.json')
CA_X_MR_WEIGHTED_HS = _load('ca-x-mr-weighted-hs.json')
CA_CAT_X_MR_X_CA_SUBVAR_HS = _load('ca-cat-x-mr-x-ca-subvar-hs.json')
CA_X_MR_HS = _load('ca-x-mr-hs.json')
MR_X_CAT_HS = _load('mr-x-cat-hs.json')
CAT_X_MR_HS = _load('cat-x-mr-hs.json')
CA_X_MR_SIG_TESTING_SUBTOTALS = _load('ca-x-mr-sig-testing-subtotals.json')
STARTTIME_X_NORDIC_COUNTRIES_X_FOOD_GROOPS = _load(
    'starttime-x-nordic-countries-x-food-groops.json',
)
FOOD_GROOPS_X_STARTTIME_X_NORDIC_COUNTRIES = _load(
    'food-groops-x-starttime-x-nordic-countries.json',
)
MR_X_CAT_X_MR_PRUNE = _load('mr-x-cat-x-mr-prune.json')
HUFFPOST_ACTIONS_X_HOUSEHOLD = _load('huffpost-actions-x-household.json')
GENDER_X_WEIGHT = _load('gender-x-weight.json')
CAT_X_CAT_PRUNING_HS = _load('cat-x-cat-pruning-hs.json')
CA_ITEMS_X_CA_CAT_X_CAT = _load('ca-items-x-ca-cat-x-cat.json')
CAT_X_MR_X_CAT = _load('cat-x-mr-x-cat.json')
CAT_X_CAT_FILTERED_POP = _load('cat-x-cat-filtered-population.json')
UNIV_MR_WITH_HS = _load('univ-mr-with-hs.json')
