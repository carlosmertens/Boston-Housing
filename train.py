# PROGRAMMER: Carlos Mertens
# DATE CREATED: (DD/MM/YY) - 20/12/18
# REVISED DATE: (DD/MM/YY) - Not revise it yet
# PURPOSE: Build an optimal supervised learner model based on a statistical analysis.
#           Load, split, prepare can calculate statistics on the data.
#           Perform metric evaluation, grid search. Train optimal model 
#
# USAGE: Use argparse Expected Call with <> indicating expected user input:
#           python check_images.py --dir <directory with data> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python train.py --dir Data/housing.csv

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
