import os
import json

thisdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(thisdir, 'cubes', 'cat-x-cat.json')) as f:
    fixt_cat_x_cat = json.load(f)
with open(os.path.join(thisdir, 'cubes', 'univariate-categorical.json')) as f:
    fixt_univariate_categorical = json.load(f)
