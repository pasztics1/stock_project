import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

import os
path = os.path.join(os.getcwd(), "data")

from main import RF,RF_boosted
from feature_engineering import add_features
from read_data import correct_format

#hyperparams
PERC_DATA_USED = 1

#initializing file names for i/o
# ask_file_name = "AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv"
# bid_file_name = "AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv"

ask_file_name = "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-26.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-26.10.2024.csv"
delta_t = 5

y_types = ["binary_classifier","percentage_change","price_movements_classification"]
#adding feature engineering to our data if it doesn't exist
for y_type in y_types:
    features_name = f'features_{y_type}delta_t{delta_t}{PERC_DATA_USED}{ask_file_name}'
    if not os.path.isfile(os.path.join(path,features_name)):
        add_features(ask_file_name,bid_file_name,PERC_DATA_USED, y_type, delta_t)
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
        'Open_BID', 'High_BID', 'Low_BID', 'Close_BID', 'Volume_BID',
        'Open_ASK', 'High_ASK', 'Low_ASK', 'Close_ASK', 'Volume_ASK', 'Mid_Price',
        'Spread', 'Volume', 'Open_Mid', 'High_Mid', 'Low_Mid', 'Close_Mid',
        'Unix_Diff_Until_Earnings', 'Month', 'Day_Type', 'Week_of_Year', 'Quarter',
        'Hour', 'Is_Open_Hour', 'Is_Close_Hour', 'Day_of_Week', 'Is_Lunch_Time',
        'Time_Since_Open', 'score', 'rating', 'SMA_3', 'SMA_6', 'SMA_12', 'SMA_5',
        'SMA_10', 'SMA_20', 'EMA_3', 'EMA_6', 'EMA_12', 'EMA_26', 'MACD',
        'Signal_Line', '%K', '%D', 'CCI', 'Return_1H', 'Return_3H', 'Return_6H',
        'RSI_long', 'RSI_short', 'Middle_Band', 'Upper_Band', 'Lower_Band',
        'High_Low', 'High_Close', 'Low_Close', 'True_Range', 'ATR', 'Volume_SMA_3',
        'Volume_SMA_6', 'OBV', 'MF_Multiplier', 'MF_Volume', 'CMF', 'Volatility',
        'Return', 'Volume_change', 'Lag_Mid_Price_1', 'Lag_Mid_Price_2',
        'Lag_Mid_Price_3', 'Lag_Mid_Price_4', 'Lag_Return_1', 'Lag_Return_2',
        'Lag_Return_3', 'Lag_Return_4', 'Return_1', 'Return_5', 'Return_10',
        'Rolling_Mean_5', 'Rolling_Std_5', 'Lagged_Volatility', 'Hammer',
        'Rolling_Skew', 'Rolling_Kurt', 'Price_Z_Score', 'Conversion_Line',
        'Base_Line', 'Leading_Span_A', 'Leading_Span_B', 'Williams_%R',
        'High_Low_Ratio', 'Close_Open_Ratio', 'Order_Imbalance', 'VPT', 'A/D_Line',
        'Support_10', 'Resistance_10', 'Support_Close_10', 'Resistance_Close_10',
        'Support_20', 'Resistance_20', 'Support_Close_20', 'Resistance_Close_20',
        'SMA5_Volume', 'RSI_Volatility'
    ]


    feature_scores = pd.DataFrame({'Feature': column_names, 'Score': scores})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    

    # Example of creating or writing to a file in the "data" directory
    output_file_path = os.path.join(path, ('scores'+features_name))  # Ensure to provide a filename here

    # Writing to the file
    feature_scores.to_csv(output_file_path)
    print(f"Feature scores for {features_name}!\n\n{feature_scores}")
