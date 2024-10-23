# username = "pasztics1"
# token = "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"
# !git clone https://{username}:{token}@github.com/pasztics1/stock_project.git

# %cd stock_project
#ctr alt enter - start cell
#alt shift num lock - cursor
#ctrl + - run code

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from main import DT,RF
from labeling import correct_file
from feature_engineering import add_features


PERC_DATA_USED = 0.15

add_features("AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv","AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv",PERC_DATA_USED)
#set the rigth directory, it's now in the data folder

#it's the ask file here bcs that's what the add_features gives back
data = correct_file(f'features_{PERC_DATA_USED}AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv') #https://numpy.org/devdocs/user/how-to-io.html


X = data.iloc[:, 1:-1].values #first row's not included, since it's date
y = data.iloc[:, -1].values
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


clf = RF()
clf.fit(X_train, y_train)



import pickle

#Saving model parameters
with open(f'save_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# To load the model later:
# with open('random_forest_model.pkl', 'rb') as f:
#     clf = pickle.load(f)


predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, predictions)
print(f'Normal accuracy : {acc}')

for i in range(50,100,5):
    predictions = clf.predict(X_test,False,i/100)
    acc = accuracy(y_test, predictions[:,0])
    print(f'{i}% certanty accuracy: {acc}')


