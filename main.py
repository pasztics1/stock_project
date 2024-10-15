#predicting; a particular stock (AAPL,TSLA,MSFT,AMZN,NVDA,Alphabet), an index, maybe an asset or making a generalized model
#newsapi.org
#yahoofinance
import pandas as pd
import numpy as np
import yfinance as yf

from labeling import correct_file
from collections import Counter
#cd C:\Users\Surface\Desktop\binary_classifier_project


#ideas:
  #improvement of feature selection
  #implementation of cross-validation


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
    def __init__(self, min_samples_split=2, max_depth = 100, n_features = 30):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features #Adding randomness by not using all the features, just a subset of them, crucial for Random forests, that's different from original ID3
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
            return Node(value=leaf_value) #that's what the "*" does
        
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False) #https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    
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
        
        for feat_idx in feat_idxs:
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
        count = Counter(y)
        value = count.most_common(1)[0][0] #https://docs.python.org/3/library/collections.html#collections.Counter
        return value
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        

        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)
        




class RF:
    def __init__(self, certanity_needed=0.6, n_trees=30,max_depth=15,min_samples_split=2, n_features=30):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.certainty_needed=certanity_needed
        self.n_features=n_features
        self.trees=[]
        
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DT(max_depth=self.max_depth,
               min_samples_split=self.min_samples_split,
               n_features=self.n_features)
            
            
            
            X_sample, y_sample = self._samples(X, y)
            tree.fit(X_sample, y_sample)
            print(f"Tree {len(self.trees)} done!\nTraining progress: {round(len(self.trees)/self.n_trees,2)}%")
            self.trees.append(tree) #appending tree to the forest
            
    def _samples(self, X, y): #what to name this
        n_samples = X.shape[0] #when calling this the first element is the n of features.
        idxs = np.random.choice(n_samples, n_samples, replace=True) #as you can see I set the replace to true, so the same info is going to be given to multiple trees
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        count = Counter(y)
        value = count.most_common(1)[0][0] #https://docs.python.org/3/library/collections.html#collections.Counter
        return value
        
    def _certainty(self, y):
        count = Counter(y)
        if min(count[True],count[False])/max(count[True],count[False])<self.certainty_needed:
            return count.most_common(1)[0][0]
        return 0
            

    
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]) #predict is a 2d array with the arrays containing each prediction for every x in our testing subset [[tree_0_prediction_0,tree_0_prediction_1,...tree_0_prediction_n],...[tree_n_prediction_0,...tree_n_prediction_n]]
        #but what we want to work with is a 2d array which contains sub arrays that have all the predictions from all the trees for just one x. So [[tree_0_prediction_0, tree_1_prediction_0,...tree_n_prediction_0],...[tree_0_pediction_n,...tree_n_prediction_n]]
        tree_preds = np.swapaxes(predictions, 0, 1) #
        predictions=np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
    
