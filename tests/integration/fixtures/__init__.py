import os
import json

thisdir = os.path.dirname(os.path.abspath(__file__))

# Bivariate Cubes
with open(os.path.join(thisdir, 'cubes', 'cat-x-cat.json')) as f:
    fixt_cat_x_cat = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'cat-x-datetime.json')) as f:
    fixt_cat_x_datetime = json.load(f)

# Univariate Cubes
with open(os.path.join(thisdir, 'cubes', 'univariate-categorical.json')) as f:
    fixt_univariate_categorical = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'voter-registration.json')) as f:
    fixt_voter_registration = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'simple-datetime.json')) as f:
    fixt_simple_datetime = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'simple-text.json')) as f:
    fixt_simple_text = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'simple-cat-array.json')) as f:
    fixt_simple_cat_array = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'cat-x-num-x-datetime.json')) as f:
    fixt_cat_x_num_x_datetime = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'simple-mr.json')) as f:
    fixt_simple_mr = json.load(f)
