import pandas as pd
import numpy as np
import os
import joblib
import re
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


# Import custom utility modules
from confident_tester import evaluate_models
from feature_evaluation import feature_evaluation
from feature_selection import select_optimal_features  
from read_data import correct_format 

# Import SMOTE and Pipeline from imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Import SHAP
import shap


# # ===========================
# # Reset Previous Training Artifacts
# # ===========================

# # Define paths for model and log directories
# model_output_path = os.path.join(os.getcwd(), "saved_models")
# log_path = os.path.join(os.getcwd(), "save_hyperparam_configs")
# optuna_db_path = "optuna_study.db"  # Adjust if your Optuna database is elsewhere

# # Remove old saved models
# if os.path.exists(model_output_path):
#     shutil.rmtree(model_output_path)
# os.makedirs(model_output_path, exist_ok=True)

# # Remove old log files
# if os.path.exists(log_path):
#     shutil.rmtree(log_path)
# os.makedirs(log_path, exist_ok=True)

# # Remove old Optuna database file
# if os.path.exists(optuna_db_path):
#     os.remove(optuna_db_path)


# ===========================
# Hyperparameter Tuning Setup
# ===========================

def objective(trial, X, y, tscv):
    param = {
        'model__objective': 'binary',
        'model__metric': ['binary_logloss', 'auc'],
        'model__boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'model__verbosity': -1,
        'model__random_state': 42,
        'model__n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'model__learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
        'model__max_depth': trial.suggest_int('max_depth', 5, 100),
        'model__num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'model__min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 500),
        'model__feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'model__lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'model__lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'model__class_weight': 'balanced'
    }

    # Only set bagging parameters if boosting_type is not 'goss'
    if param['model__boosting_type'] != 'goss':
        param['model__bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.4, 1.0)
        param['model__bagging_freq'] = trial.suggest_int('bagging_freq', 1, 20)
        param['model__bagging_seed'] = trial.suggest_int('bagging_seed', 0, 100)

    # Rest of your code remains the same


    # Define the pipeline with scaling, SMOTE, and the model using ImbPipeline
    pipeline = ImbPipeline([
        ('feature_names', FunctionTransformer(store_feature_names, validate=False)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', lgb.LGBMClassifier())
    ])

    # Set the parameters in the pipeline
    pipeline.set_params(**param)

    roc_auc_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]

        # Fit the pipeline
        pipeline.fit(
            X_train, y_train,
            model__eval_set=[(X_valid, y_valid)],
            model__eval_metric='binary_logloss',
            model__callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

        # Set feature names after the fit
        pipeline.feature_names = store_feature_names.feature_names

        y_pred_prob = pipeline.predict_proba(X_valid)[:, 1]
        roc_auc = roc_auc_score(y_valid, y_pred_prob)
        roc_auc_scores.append(roc_auc)

    return np.mean(roc_auc_scores)



def optimize_lightgbm_hyperparameters(X, y, n_trials):
    """
    Optimize LightGBM hyperparameters using Optuna with nested cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y, tscv), n_trials=n_trials)

    print("\nBest hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key.replace('model__', '')}: {value}")
    print(f"Best average ROC AUC: {study.best_value:.4f}")

    return study.best_params

# Store feature names from the input data if available
def store_feature_names(X, y=None):
    if hasattr(X, 'columns'):
        store_feature_names.feature_names = X.columns.tolist()
    return X

def sanitize_filename(s):
    return re.sub(r'[\\/:"*?<>|]+', '_', s)

# ===========================
# Processing Function
# ===========================
import datetime
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

def process_delta_t(delta_t, PERC_DATA_USED, ask_file_name, bid_file_name, model_type, y_type,
                   top_n_values, correlation_threshold, data_path, model_output_path,
                   hyperparams, plotting=False):
    """
    Processes a single delta_t value with a given hyperparameter set:
    - Feature evaluation and selection
    - Model training
    - Model saving
    - Logging hyperparameters and timestamps
    Returns a dictionary of results.
    """
    print(f"\n==============================")
    print(f"Processing delta_t = {delta_t} hours or {round(delta_t / 6, 2)} days")
    print(f"==============================\n")
    
    # Initialize a local result dictionary
    local_result = {}

    # Step 1: Feature Evaluation
    print("Step 1: Feature Evaluation")
    feature_scores_file = feature_evaluation(
        ask_file_name=ask_file_name,
        bid_file_name=bid_file_name,
        PERC_DATA_USED=PERC_DATA_USED,
        delta_t=delta_t,
        model_type=model_type,
        y_types=y_type
    )

    # Step 2: Feature Selection
    print("\nStep 2: Feature Selection")
    features_name = f'features_{y_type[0]}delta_t{delta_t}{PERC_DATA_USED}{ask_file_name}'
    
    included_features = select_optimal_features(
        feature_scores_file=feature_scores_file,
        dataset_file=features_name,
        top_n_values=top_n_values,
        model=model_type,
        correlation_threshold=correlation_threshold,
        plot=plotting
    )

    # Step 3: Data Loading and Preprocessing
    print("\nStep 3: Data Loading and Preprocessing")
    data = correct_format(features_name)
    # We only train on the most important features
    X = data[included_features].values
    y = data['y'].values  # Assuming 'y' is the target variable
    # Correct format
    X = X.astype(np.float64)
    y = y.astype(np.int8)
    print(f"Selected Features Shape: {X.shape}")

    # Step 4: Train the Model
    print("\nStep 4: Training LightGBM Model with Given Hyperparameters")
    
    # Define the pipeline with scaling, SMOTE, and the model
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', LGBMClassifier(**hyperparams))
    ])

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    roc_auc_scores = []
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        print(f"\n--- Fold {fold} ---")
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        
        pipeline.fit(
            X_train, y_train,
            model__eval_set=[(X_valid, y_valid)],
            model__eval_metric='binary_logloss',
            model__callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

        # Predict and calculate ROC AUC for the validation set
        y_pred_prob = pipeline.predict_proba(X_valid)[:, 1]
        roc_auc = roc_auc_score(y_valid, y_pred_prob)
        roc_auc_scores.append(roc_auc)
        print(f"Fold {fold} ROC AUC: {roc_auc:.4f}")

    avg_roc_auc = np.mean(roc_auc_scores)
    print(f"Average ROC AUC: {avg_roc_auc:.4f}")

    # Step 5: Save the Model
    original_feature_names = included_features  # Assuming `included_features` are the market indicators

    # Add the feature names to the pipeline
    pipeline.named_steps['model'].original_feature_names = original_feature_names
    print("\nSaving the Trained Model")
    model_filename = f'final_lightgbm_model_delta_t_{delta_t}.pkl'
    model_filepath = os.path.join(model_output_path, model_filename)
    joblib.dump(pipeline, model_filepath)
    print(f"Model saved at {model_filepath}")


    # Step 6: Log Hyperparameters and Timestamps
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparam_dir = os.path.join(os.getcwd(), "save_hyperparam_configs")
    os.makedirs(hyperparam_dir, exist_ok=True)
    unique_hyperparam_filename = f'hyperparams_delta_t_{delta_t}_{timestamp}.json'
    log_file = os.path.join(hyperparam_dir, unique_hyperparam_filename)

    print("\nStep 6: Logging Hyperparameters and Timestamps")
    log_entry = {
        'delta_t': delta_t,
        'hyperparameter_set': hyperparams,
        'start_time': timestamp,
        'model_path': model_filepath
    }
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=4)
    print(f"Logged hyperparameters and timestamp in '{log_file}'")

    # Step 7: Return Results
    local_result = {
        'hyperparameter_set': hyperparams,
        'average_roc_auc': avg_roc_auc,
        'model_path': model_filepath,
        'timestamp': timestamp
    }
    return local_result

    # except Exception as e:
    #     print(f"Error processing delta_t={delta_t}: {e}")
    #     # Ensure all keys are present even in case of error
    #     local_result = {
    #         'hyperparameter_set': hyperparams,
    #         'average_roc_auc': None,
    #         'model_path': None,
    #         'timestamp': None,
    #         'error': str(e)
    #     }
    #     return local_result


# ===========================
# Main Execution Workflow
# ===========================

# ===========================
# Configuration Parameters
# ===========================

import itertools
import random

# Define hyperparameter ranges#

# Adjusted hyperparameter grid

hyperparameter_grid = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 50, 100, 150],
    'max_depth': [10, 20, 30, -1],
    'min_data_in_leaf': [20, 50, 100, 200],
    'feature_fraction': [0.6, 0.8, 1.0],
    'lambda_l1': [0, 0.1, 1.0, 10.0],
    'lambda_l2': [0, 0.1, 1.0, 10.0],
    'min_gain_to_split': [0, 0.1, 0.5, 1.0],
    'max_bin': [255, 500, 1000]
}

# Only include bagging parameters for 'gbdt' and 'dart'
bagging_params = {
    'bagging_fraction': [0.6, 0.8, 1.0],
    'bagging_freq': [5, 10, 15],
    'bagging_seed': [42, 100, 999],
}

# Generate combinations considering the boosting type
hyperparameter_list = []
for boosting_type in hyperparameter_grid['boosting_type']:
    grid = {key: hyperparameter_grid[key] for key in hyperparameter_grid if key != 'boosting_type'}
    grid['boosting_type'] = [boosting_type]
    
    if boosting_type != 'goss':
        # Include bagging parameters
        for key in bagging_params:
            grid[key] = bagging_params[key]
    else:
        # Exclude bagging parameters
        pass
    
    # Generate all combinations for this boosting type
    combinations = list(itertools.product(*(grid[key] for key in grid)))
    for combo in combinations:
        params = dict(zip(grid.keys(), combo))
        hyperparameter_list.append(params)


# Optional: Shuffle the list to ensure randomness
random.seed(42)
random.shuffle(hyperparameter_list)



# Data Parameters
PERC_DATA_USED = 0.8
#delta_t_values = [12, 15, 16, 17, 18, 19, 25, 30, 32, 35, 36, 38, 40]  # in hours
n_trials=500

y_type = ["binary_classifier"]
# File Parameters
ask_file_name = "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-31.10.2024.csv" 
# Feature Selection Parameters

# top_n_values = [8,9,10,11,12,13,14,15,16,18,20,25,30]

correlation_threshold = 0.6

# Model Parameters

model_type = "lightgbm"  # Options: 'random_forest', 'xgboost', 'lightgbm'

# Output Paths
data_path = os.getcwd()
model_output_path = os.path.join(data_path, "saved_models")  # Directory to save trained models
os.makedirs(model_output_path, exist_ok=True)


# HYPERPARAMS FOR MODEL EVALUATION
TEST_DATA_USED = 0.2
threshold_optimizer = "pr_curve" #can be "iterative" & "pr_curve"
save_metrics = 'write' #can be write and add, if write it creates a new txt everytime, if add, it adds to the txt.

N_TOP_MODELS = 5 #number of top models saved 
#

# Initialize a list to store results
results = []
delta_t_values_list = [[12,15,16,17,18,19,25,30,32,35,36,38,40],[5,10,15,20,25,30,35,40], [26,27,28,29,30,31,32,33,34], [11,12,13,14,15,16,17,18,19,21,23]]
top_n_values_list = [[7,8,9,10,11,12],[13,14,15,16,17],[18,19,20,21,22],[23,24,25,26,27]]


# Iterate through each hyperparameter set
for top_n_values in top_n_values_list:
    for delta_t_values in delta_t_values_list:
        for hp_idx, hyperparams in enumerate(hyperparameter_list, 1):
            print(f"\n==============================")
            print(f"Starting Hyperparameter Set {hp_idx}/{len(hyperparameter_list)}")
            print(f"==============================\n")
            
            # Record the start time for this hyperparameter set
            hp_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            
            # Iterate through each delta_t value
            for dt_idx, dt in enumerate(delta_t_values, 1):
                print(f"\n--- Training model {dt_idx}/{len(delta_t_values)} for delta_t={dt} hours ---")
                result = process_delta_t(
                    delta_t=dt,
                    PERC_DATA_USED=PERC_DATA_USED,
                    ask_file_name=ask_file_name,
                    bid_file_name=bid_file_name,
                    model_type=model_type,
                    y_type=y_type,
                    top_n_values=top_n_values,
                    correlation_threshold=correlation_threshold,
                    data_path=data_path,
                    model_output_path=model_output_path,
                    hyperparams=hyperparams,
                    plotting=False
                )
                results.append(result)
                
                # Optional: Save interim results to prevent data loss
                interim_results_csv = os.path.join(model_output_path, 'interim_model_training_results.csv')
                pd.DataFrame(results).to_csv(interim_results_csv, index=False)
                print(f"Interim results saved to '{interim_results_csv}'")
            
            # Record the finish time for this hyperparameter set
            hp_finish_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nHyperparameter Set {hp_idx} started at {hp_start_time} and finished at {hp_finish_time}")
            
            # ===========================
            # Step 4: Evaluate Models for Current Hyperparameter Set
            # ===========================
            print("\n==============================")
            print("Evaluating Models for Current Hyperparameter Set")
            print("==============================\n")
            
            # Assuming evaluate_models() can accept a list of model paths
            # Modify evaluate_models to accept specific models to evaluate
            # Here's how you might adjust the function call:
            
            evaluate_models(
                ask_file_name=ask_file_name,
                bid_file_name=bid_file_name,
                TEST_DATA_USED=TEST_DATA_USED,
                delta_t_values=delta_t_values,  # All delta_t values
                threshold_optimizer=threshold_optimizer,  # Options: "iterative" & "pr_curve"
                save_metrics=save_metrics,  # Options: 'write' or 'add'
                N_TOP_MODELS=N_TOP_MODELS,  # Number of top models to save
            )
            
            print(f"Evaluation completed for Hyperparameter Set {hp_idx}.\n")
    
# After all iterations
print("\n==============================")
print("All Models Trained and Evaluated")
print("==============================\n")
results_df = pd.DataFrame(results)
print(results_df)

# Save final results to CSV for further analysis
final_results_csv = os.path.join(model_output_path, 'final_model_training_results.csv')
results_df.to_csv(final_results_csv, index=False)
print(f"\nAll results saved to '{final_results_csv}'")




