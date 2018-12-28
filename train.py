# PROGRAMMER: Carlos Mertens
# DATE CREATED: (DD/MM/YY) - 20/12/18
# REVISED DATE: (DD/MM/YY) - Not revise it yet
# PURPOSE: Build an optimal supervised learner model based on a statistical analysis.
#           Load, split, prepare can calculate statistics on the data.
#           Perform metric evaluation, grid search. Train optimal model.
#           Test model with the following testing clients:
#               client 1 = [5, 17, 15]
#               client 2 = [4, 32, 22]
#               client 3 = [8, 3, 12]
#
# USAGE: 
#   Example call:
#    python train.py

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor


def main():
    """Start the program and call the functions."""
    data, features, prices = get_data('Data/housing.csv')

    # Shuffle and split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=42)

    print("\n** 20% of the dataset has been split for testing.")
    # Fit the training data to the model using grid search
    model = fit_model(X_train, y_train)

    # Print the value for 'max_depth'
    print("\n** Parameter 'max_depth' is {} for the optimal model.".format(model.get_params()['max_depth']))

    print("\n** Test model with client1 = [5, 17, 15], client2 = [4, 32, 22], client3 = [8, 3, 12]\n")

    # Produce a matrix for client data to be tested
    client_data = [[5, 17, 15],
                   [4, 32, 22],
                   [8, 3, 12]]

    # Print predictions
    for i, price in enumerate(model.predict(client_data)):
        print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

    # Print trials with different training and testing datapoints
    print("\n** Run function ten times with different training and testing sets:")

    predict_trials(features, prices, fit_model, client_data)



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


def performance_metric(y_true, y_predict):
    """ Calculates and returns the coefficient of determination R2.
    
    R2 captures the percentage of squared correlation between the predicted 
    and actual values of the targets.
    Parameters:
     y_true - Array, the actual target values
     y_predict - Array, the predicted target values
    Return:
     score - Float, the r2 metric score between 0 and 1 to be interpreted as 
             a percentage between 0 and 100
    """
    # Calculate the performance score between true target values and predicted values
    score = r2_score(y_true, y_predict)

    return score


def fit_model(X, y):
    """ Performs grid search and train model.
    
    Grid search is performed over the 'max_depth' parameter from 1 to 10 for a 
    decision tree regressor training.
    Parameters:
     X - Array, features from our data to be used for training
     y - Array, target values for the features on the training
    Return:
     grid.best_estimator_ - Optimal model trained with parameters from the grid search
    """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()
    regressor.fit(X, y)

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1, 11)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def predict_trials(X, y, fitter, data):
    """Perform trials of fitting and predicting data."""
    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2, random_state=k)

        # Fit the data
        reg = fitter(X_train, y_train)

        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)

        # Result
        print("Trial {}: ${:,.2f}".format(k + 1, pred))

    # Display price range
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))


# Call main function to run the program
if __name__ == '__main__':
    main()
