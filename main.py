#predicting; a particular stock (AAPL,TSLA,MSFT,AMZN,NVDA,Alphabet), an index, maybe an asset or making a generalized model
#newsapi.org
#yahoofinance
import pandas as pd
import numpy as np
from collections import Counter

#these are needed for speeding up the fitting process
from tqdm import tqdm #needed for graphing
from multiprocessing import shared_memory #needed for shared memory
from joblib import Parallel, delayed #needed for paralel processing

#ideas:
  #improvement of feature selection
  #implementation of cross-validation

#HYPERPARAMETERS#

N_TREES = 100    #30 This was modified
MAX_DEPTH = 5   #10 
MIN_SAMPLES_SPLIT = 5  #2 this was modified
N_FEATURES = 7

#



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
    def __init__(self, min_samples_split=MIN_SAMPLES_SPLIT, max_depth = MAX_DEPTH, n_features = N_FEATURES):
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
    def __init__(self, n_trees=N_TREES,max_depth=MAX_DEPTH,min_samples_split=MIN_SAMPLES_SPLIT, n_features=N_FEATURES):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_features
        self.trees=[]
        
        
    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            
            tree = DT(max_depth=self.max_depth,
               min_samples_split=self.min_samples_split,
               n_features=self.n_features)
            
            
            
            X_sample, y_sample = self._samples(X, y)
            tree.fit(X_sample, y_sample)
            print(f"Tree {len(self.trees)+1} done!\nTraining progress: {round((1+len(self.trees))/self.n_trees,2)*100}%")
            self.trees.append(tree) #appending tree to the forest
            
    def _samples(self, X, y): #what to name this
        n_samples = X.shape[0] #when calling this the first element is the n of features.
        idxs = np.random.choice(n_samples, n_samples, replace=True) #as you can see I set the replace to true, so the same info is going to be given to multiple trees
        return X[idxs], y[idxs]
    
    
    def _aggregate_predictions(self, predictions, certanity_perc): #if the tree's not confident enought in the choices, it doesn't say a prediction.
        count = Counter(predictions)
        total_votes = self.n_trees
        most_common_label, num_votes = count.most_common(1)[0]

        certainty = num_votes/total_votes
        if certainty >= certanity_perc:
            return [most_common_label,certainty]

        else:
            return [None,certainty]
            
        
    def predict(self, X, certainty_perc=0.5):
        #predict is a 2d array with the arrays containing each prediction for every x in our testing subset [[tree_0_prediction_0,tree_0_prediction_1,...tree_0_prediction_n],...[tree_n_prediction_0,...tree_n_prediction_n]]
        #but what we want to work with is a 2d array which contains sub arrays that have all the predictions from all the trees for just one x. So [[tree_0_prediction_0, tree_1_prediction_0,...tree_n_prediction_0],...[tree_0_pediction_n,...tree_n_prediction_n]]

        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1) #
        
        return np.array([self._aggregate_predictions(pred,certainty_perc) for pred in tree_preds]) #returning filtered predictions


    
    
from joblib import Parallel, delayed #needed for paralell processing


class RF_boosted:
    def __init__(self, n_trees=N_TREES, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, n_features=N_FEATURES, n_jobs=-1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.n_jobs = n_jobs
        self.trees = []

    def _build_tree(self, X_shm, y_shm, X_shape, y_shape):
        # Access the shared memory arrays
        existing_X = np.ndarray(X_shape, dtype=np.float64, buffer=X_shm.buf)
        existing_y = np.ndarray(y_shape, dtype=np.int64, buffer=y_shm.buf)
        
        # Create a DT instance and perform bootstrap sampling
        tree = DT(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
        X_sample, y_sample = self._samples(existing_X, existing_y)
        tree.fit(X_sample, y_sample)
        return tree

    def fit(self, X, y):
        # Create shared memory blocks for X and y
        shm_X = shared_memory.SharedMemory(create=True, size=X.nbytes)
        shm_y = shared_memory.SharedMemory(create=True, size=y.nbytes)

        # Create memory-mapped arrays pointing to the shared memory
        shared_X = np.ndarray(X.shape, dtype=X.dtype, buffer=shm_X.buf)
        shared_y = np.ndarray(y.shape, dtype=y.dtype, buffer=shm_y.buf)

        # Copy data into shared memory
        np.copyto(shared_X, X)
        np.copyto(shared_y, y)

        # Parallel building of trees with tqdm progress tracking
        with tqdm(total=self.n_trees, desc="Building Trees") as pbar:
            self.trees = Parallel(n_jobs=self.n_jobs)(
                delayed(self._build_tree)(shm_X, shm_y, X.shape, y.shape) for _ in range(self.n_trees)
            )
            for _ in range(self.n_trees):
                pbar.update(1)

        # Close and unlink shared memory
        shm_X.close()
        shm_X.unlink()
        shm_y.close()
        shm_y.unlink()

    def _samples(self, X, y):
        # Bootstrap sampling directly from shared memory array
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X, certainty_perc=0.5):
        # Predict using the trained trees and aggregate results
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._aggregate_predictions(pred, certainty_perc) for pred in tree_preds])

    def _aggregate_predictions(self, predictions, certanity_perc): #if the tree's not confident enought in the choices, it doesn't say a prediction.
        count = Counter(predictions)
        total_votes = self.n_trees
        most_common_label, num_votes = count.most_common(1)[0]

        certainty = num_votes/total_votes
        if certainty >= certanity_perc:
            return [most_common_label,certainty]

        else:
            return [None,certainty]

