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


from main import DT,RF
from read_data import correct_format
from feature_engineering import add_features
PERC_DATA_USED = 1

ask_file_name = "AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv"
features_name = f'features_{PERC_DATA_USED}{ask_file_name}'

if not os.path.isfile(os.path.join(path,features_name)): #check if the file already exists
    add_features(ask_file_name,bid_file_name,PERC_DATA_USED)
else:
    print(f"{features_name} already exists!")


#it's the ask file here bcs that's what the add_features gives back
data = correct_format(f'features_{PERC_DATA_USED}AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv') #https://numpy.org/devdocs/user/how-to-io.html

X = data.iloc[:, 1:-1].values #first row's not included, since it's date
y = data.iloc[:, -1].values

X = X.astype(np.float64)
y = y.astype(np.int64)

print(f'Actual data: true: {np.sum(y==1)} false: {np.sum(y==0)}\n')

def accuracy(y_test, y_pred):
    print(f'Before: true: {np.sum(y_pred==1)}, false: {np.sum(y_pred==0)}, none: {np.count_nonzero(y_pred==None)}')
        
    #remove None before calculating accuracy
    mask = np.array([x is not None for x in y_pred])

    y_pred_filtered = y_pred[mask]
    y_test_filtered = y_test[mask]


    print(f'Ratio of confident answers {len(y_pred_filtered)/len(y_test)}')

    print(f'After: true {np.sum(y_pred_filtered==1)}, false: {np.sum(y_pred_filtered==0)}, none: {np.count_nonzero(y_pred_filtered==None)}') 
    return np.sum(y_test_filtered == y_pred_filtered) / len(y_test_filtered)



all_predictions = []
certaintys = [i/100 for i in range(50,100,5)]

max_accuracy = 0
setup = ''

#be sure to include actual names, or automatize the process
#files = ['save_model.pkl','5prec_20tree_10depth_057.pkl','20prec_20tree_10depth_056.pkl','50prec_20tree_10depth_057.pkl','100perc_20tree_10depth_05677.pkl']

files = ['15prec_250tree_15depth.pkl']

for i in range(len(files)):
    all_predictions.append([])
    with open(files[i], 'rb') as f:
        clf = pickle.load(f)
        for certainty in certaintys:
            predictions = clf.predict(X,certainty)
            all_predictions[i].append(predictions)
    
for i in range(len(all_predictions)):
    print(f"\nResults for model: {files[i]}\n")

    for j in range(len(certaintys)):
            acc = accuracy(y, all_predictions[i][j][:,0])
            print(f'{certaintys[j]} certanty accuracy:', sep='')
            print(acc)
            
            
            if acc>max_accuracy:
                setup = f"{files[i]}{certaintys[j]}"
                max_accuracy = acc


            all_predictions[i][j] = all_predictions[i][j][np.array([x is not None for x in all_predictions[i][j][:,0]])]
            print(f'Average certanity: {np.mean(all_predictions[i][j][:,1])}')
    

print(f'\n\n!!Max!!\nAccuracy: {max_accuracy}\nSetup: {setup}')

#Check if overfitting
confident_indices = np.where(all_predictions[0][0][:,0]>0.9)
np.savetxt(f'{PERC_DATA_USED}_{files[0]}confident_indices.txt', confident_indices, fmt='%d')






