import pandas as pd # type: ignore
import numpy as np
import os
import pickle
from feature_engineering import add_features

path = os.path.join(os.getcwd(), "data")

bid_filename = "AAPL.USUSD_Candlestick_1_M_BID_07.10.2024-07.10.2024.csv"
ask_filename = "AAPL.USUSD_Candlestick_1_M_ASK_07.10.2024-07.10.2024.csv"
# df_bid = pd.read_csv(os.path.join(path,"AAPL.USUSD_Candlestick_1_s_BID_07.10.2024-07.10.2024.csv"))
# df_ask = pd.read_csv(os.path.join(path,"AAPL.USUSD_Candlestick_1_s_ASK_07.10.2024-07.10.2024.csv"))
df_bid = pd.read_csv(os.path.join(path,bid_filename))
df_ask = pd.read_csv(os.path.join(path,ask_filename))

add_features(ask_filename,bid_filename,1,True)
df_input = pd.read_csv(os.path.join(path,"features_"+ask_filename))


X = df_input.iloc[:,1:-1].values
y = df_input.iloc[:,-1].values

print(X,y)

predictions = []
files = ['5prec_20tree_10depth_057.pkl','20prec_20tree_10depth_056.pkl','50prec_20tree_10depth_057.pkl','100perc_20tree_10depth_05677.pkl']
# To load the model later:
for file in files:
    with open(file, 'rb') as f:
        clf = pickle.load(f)
        predictions.append(clf.predict(X))
    

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


for i in range(len(predictions)):
    print(accuracy(y, predictions[i]))
    




total = 0



# #simulating the trades
# for i in range(len(X)):
#     if predictions[i]: #go long
        
    
#     else: #go short
        

