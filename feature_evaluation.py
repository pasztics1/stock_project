import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

import os
path = os.path.join(os.getcwd(), "data")

from main import RF,RF_boosted
from feature_engineering import add_features
from read_data import correct_format

#hyperparams
PERC_DATA_USED = 0.01

#initializing file names for i/o
ask_file_name = "AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv"
features_name = f'features_{PERC_DATA_USED}{ask_file_name}'

#adding feature engineering to our data if it doesn't exist
if not os.path.isfile(os.path.join(path,features_name)):
    add_features(ask_file_name,bid_file_name,PERC_DATA_USED)
else:
    print(f"{features_name} already exists!")

#https://numpy.org/devdocs/user/how-to-io.html
data = correct_format(features_name)

X = data.iloc[:, 1:-1].values #first row's not included, since it's date
y = data.iloc[:, -1].values

X = X.astype(np.float64)
y = y.astype(np.int64)


# For classification tasks
selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features

# For regression tasks, use f_regression
# selector = SelectKBest(score_func=f_regression, k=10)

# Fit the selector to the data
selector.fit(X, y)

# Get the scores for each feature
scores = selector.scores_

# Create a dataframe for visualization
column_names = [
    'Open_BID', 'High_BID', 'Low_BID', 'Close_BID', 
    'Open_ASK', 'High_ASK', 'Low_ASK', 'Close_ASK', 'Volume', 
    'Mid_Price', 'Spread', 'Hour', 'Month', 'Day_Type', 
    'Unix_Diff_Until_Earnings', 'score', 'rating', 'SMA_5', 
    'SMA_10', 'Volatility', 'Return', 'Volume_change', 
    'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_4', 
    'Lag_Close_5', 'Lag_Volume_1', 'Lag_Volume_2', 'Lag_Volume_3', 
    'Lag_Volume_4', 'Lag_Volume_5', 'Lag_Return_1', 'Lag_Return_2', 
    'Lag_Return_3', 'Lag_Return_4', 'Lag_Return_5', 'Support_10', 
    'Resistance_10', 'Support_Close_10', 'Resistance_Close_10', 
    'Support_20', 'Resistance_20', 'Support_Close_20', 
    'Resistance_Close_20', 'RSI'
]

feature_scores = pd.DataFrame({'Feature': column_names, 'Score': scores})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)

print(feature_scores)
