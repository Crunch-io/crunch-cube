'''Home of the fixtures for Crunch Cube integration tests.

The tests are considered 'integration', since they need to load the fixtures
from the files, however they're very close to unit tests. The reason for using
the fixtures is better visibility of the query data, and also keeping the
source files relatively clean.
'''
import os
import json

cubes_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'cubes')

# Bivariate Cubes
with open(os.path.join(cubes_directory, 'cat-x-cat.json')) as f:
    fixt_cat_x_cat = json.load(f)
with open(os.path.join(cubes_directory, 'cat-x-datetime.json')) as f:
    fixt_cat_x_datetime = json.load(f)

# Univariate Cubes
with open(os.path.join(cubes_directory, 'univariate-categorical.json')) as f:
    fixt_univariate_categorical = json.load(f)
with open(os.path.join(cubes_directory, 'voter-registration.json')) as f:
    fixt_voter_registration = json.load(f)
with open(os.path.join(cubes_directory, 'simple-datetime.json')) as f:
    fixt_simple_datetime = json.load(f)
with open(os.path.join(cubes_directory, 'simple-text.json')) as f:
    fixt_simple_text = json.load(f)
with open(os.path.join(cubes_directory, 'simple-cat-array.json')) as f:
    fixt_simple_cat_array = json.load(f)
with open(os.path.join(cubes_directory, 'cat-x-num-x-datetime.json')) as f:
    fixt_cat_x_num_x_datetime = json.load(f)
with open(os.path.join(cubes_directory, 'simple-mr.json')) as f:
    fixt_simple_mr = json.load(f)
with open(os.path.join(cubes_directory, 'cat-x-mr.json')) as f:
    fixt_cat_x_mr = json.load(f)
with open(os.path.join(cubes_directory,
                       'econ-gender-x-ideology-weighted.json')) as f:
    fixt_econ_gender_x_ideology_weighted = json.load(f)
with open(os.path.join(cubes_directory,
                       'cat-x-cat-german-weighted.json')) as f:
    fixt_cat_x_cat_german_weighted = json.load(f)
