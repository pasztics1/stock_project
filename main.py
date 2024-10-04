#predicting; a particular stock (AAPL,TSLA,MSFT,AMZN,NVDA,Alphabet), an index, maybe an asset or making a generalized model
#newsapi.org
#yahoofinance
import pandas as pd
import numpy as np
import yfinance as yf

from collections import Counter



# df=pd.read_csv('AAPL_5y_60min.csv',sep=";")
# print(df.head()) #wtf is adj close?


# for i in range(1, len(df)-1):
#     if df["Close"][i+1] > df["Close"][i]: 
#         df["Higher"][i] = 1
        
# df.to_csv('corrected.csv', index=False)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None): #By using the asterisk we always have to enter the value when calling the class
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None #root node is going to have None as a value
        
class DT:
    def __init__(self, min_samples_split=2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features #Adding randomness by not using all the features, just a subset of them, crucial for Random forests
        self.root = None
        
    def fit(self, X, y):
        
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features) #making sure that the n_features do not exceed the num. of features that we have.
        self.root = self._grow_tree(X,y)
         
    def _grow_tree(self, X, y, depth=0): #going to be run recursively (remember the tutorial)
        n_samples, n_feats = X.shape #shape returns 2 values and this way you can give those to 2 variables
        n_labels = len(np.unique(y))
        
        
        # checking for stopping criteria   
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value) #what the "*" does
        
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
    
        #best split based on IG
        best_feature, best_thresh = self._best_split(X, y, feat_idxs) #feat_idx is the features I want to include when creating a new split, that's where randomness is created
        
        
        #create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature],best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1) 
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)
        
        
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs: #maybe indexing instead of nesting
            X_column = X[:, feat_idx] #that's how indexing in np looks
            thresholds = np.unique(X_column)
            
            for thr in thresholds:
                #calc. IG
                  
                gain = self._info_gain(y, X_column, thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold
    
    
    def _info_gain(self, y, X_column, threshold):
        #parent entropy
        parent_entropy = self._entropy(y)
        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 
        
        #calculate the entropy of children (and taking their weighted avg.)
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r     
        
        
        #calculate IG 
        information_gain = parent_entropy - child_entropy
        return information_gain
        
    def _split(self, X_coulmn, split_thresh):
        left_idxs = np.argwhere(X_coulmn<=split_thresh).flatten() #as in excel, and flatten makes an n dimensional list "flat", so 1d
        right_idxs = np.argwhere(X_coulmn>split_thresh).flatten()
        return left_idxs, right_idxs
        
        
    def _entropy(self, y):
        hist = np.bincount(y) #create a histogram
        ps = hist/len(y) #yeah, that's something you can do in np (https://numpy.org/doc/2.0/user/absolute_beginners.html)
        return -np.sum([p * np.log(p) for p in ps if p>0])
        
        
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0] #https://docs.python.org/3/library/collections.html#collections.Counter
        return value
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)
        

