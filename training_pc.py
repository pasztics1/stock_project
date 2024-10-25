# username = "pasztics1"
# token = "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"
# !git clone https://{username}:{token}@github.com/pasztics1/stock_project.git

# %cd stock_project
#ctr alt enter - start cell
#alt shift num lock - cursor
#ctrl + - run code
import os
path = os.path.join(os.getcwd(), "data")

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle


from main import RF,
from feature_engineering import add_features
from read_data import correct_format

#hyperparams
PERC_DATA_USED = 0.15

#initializing file names for i/o
ask_file_name = "AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv"
features_name = f'features_{PERC_DATA_USED}{ask_file_name}'

#adding feature engineering to our data
add_features(ask_file_name,bid_file_name,PERC_DATA_USED)

#https://numpy.org/devdocs/user/how-to-io.html
data = correct_format(features_name)

X = data.iloc[:, 1:-1].values #first row's not included, since it's date
y = data.iloc[:, -1].values
print(data.head())


#splitting the data into training and test portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

#training the data
clf = RF()
clf.fit(X_train, y_train)

#Saving model parameters
with open(f'save_model.pkl', 'wb') as f:
    pickle.dump(clf, f)



