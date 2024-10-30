#ctr alt enter - start cell
#alt shift num lock - cursor
#ctrl + - run code
import os
path = os.path.join(os.getcwd(), "data")

from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle


from main import RF,RF_boosted
from feature_engineering import add_features
from read_data import correct_format
from feature_evaluation import feature_evaluation
from feature_selection import select_optimal_features

#hyperparams
PERC_DATA_USED = 0.05
delta_t = 5
use_xgboost = False
y_type = "binary_classifier"

#initializing file names for i/o
ask_file_name = "AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv"
features_name = f'features_{y_type}delta_t{delta_t}{PERC_DATA_USED}{ask_file_name}'

feature_scores_file = feature_evaluation(ask_file_name,bid_file_name,PERC_DATA_USED, delta_t, use_xgboost)
included_features = select_optimal_features(feature_scores_file,features_name,[10,12,15,16,18,20,25,30])



data = correct_format(features_name)

#we only train on the most important features
X = data[included_features].iloc[:,:].values
#X = data.iloc[:, :-1].values #first row's not included, since it's date

y = data.iloc[:, -1].values

X = X.astype(np.float64)
print(X.shape)
y = y.astype(np.int64)



#splitting the data into training and test portions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

#training the data using my model from main.py 

clf = RF_boosted()
clf.fit(X_train, y_train)

#training the data using an optimized, pre defined model



#Saving model parameters
with open(f'save_model.pkl', 'wb') as f:
    pickle.dump(clf, f)



# Define cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=1234)  # 5-fold cross-validation

# Function to train and evaluate on a single fold
def evaluate_fold(train_index, test_index):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    # Train model on the current fold
    clf = RF_boosted()  # Create a new instance of the model for each fold
    clf.fit(X_train_cv, y_train_cv)
    
    # Make predictions on the validation fold
    predictions = clf.predict(X_test_cv)[:, 0]  # Predictions might include certainty, so we only take the first column
    
    # Calculate accuracy (ignoring None predictions if applicable)
    mask = np.array([x is not None for x in predictions])
    y_pred_filtered = predictions[mask]
    y_test_filtered = y_test_cv[mask]
    
    accuracy = np.sum(y_pred_filtered == y_test_filtered) / len(y_test_filtered)
    return accuracy

# Perform cross-validation in parallel
cv_accuracies = Parallel(n_jobs=-1)(delayed(evaluate_fold)(train_idx, test_idx) for train_idx, test_idx in kf.split(X))

# Calculate and print the average cross-validation accuracy
cv_accuracy = np.mean(cv_accuracies)
print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")