from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from main import DT

'''
data = datasets.load_breast_cancer()
X,y = data.data, data.target
print(data)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


clf = DT()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

    
acc = accuracy(y_test, predictions)
print(acc)
'''

data = pd.read_csv('corrected.csv') #https://numpy.org/devdocs/user/how-to-io.html
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
print(data.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


clf = DT()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

    
acc = accuracy(y_test, predictions)

print(acc)