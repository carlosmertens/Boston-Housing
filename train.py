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


def get_data(path):
    """Load csv file and get features and target.
    
    Function to load csv file with the Boston housing data using Pandas. 
    Separate price column to be the target and the other columns to be the features.
    
    Parameters:
     path - Full path to the csv file with the dataset
    Returns:
     data - Full dataframe
     prices - Column with datapoints to be the targets
     features - Dataframe without the target column to be the features.
    """
    # Load the Boston housing dataset and separate features and target values
    data = pd.read_csv(path)
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)

    # Print shape of dataframe
    print("** Boston housing dataset has {} datapoints with {} variables each.".format(*data.shape))

    return data, features, prices


data, features, prices = get_data('Data/housing.csv')

# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=42)

print("\n** 20% of the dataset has been split for testing.")
