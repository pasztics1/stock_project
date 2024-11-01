import pandas as pd
import numpy as np
import os
import joblib
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.preprocessing import StandardScaler

# Import custom utility modules
from feature_evaluation import feature_evaluation
from feature_selection import select_optimal_features  
from read_data import correct_format 

# Import SMOTE and Pipeline from imblearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Import SHAP
import shap

# ===========================
# Hyperparameter Tuning Setup
# ===========================

def objective(trial, X, y, tscv):
    """
    Objective function for Optuna to optimize LightGBM hyperparameters within a Pipeline.
    """
    param = {
        'model__objective': 'binary',
        'model__metric': 'binary_logloss',
        'model__boosting_type': 'gbdt',
        'model__verbosity': -1,
        'model__random_state': 42,
        'model__n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'model__learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
        'model__max_depth': trial.suggest_int('max_depth', 5, 100),
        'model__num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'model__min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 500),
        'model__feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'model__bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'model__bagging_freq': trial.suggest_int('bagging_freq', 1, 20),
        'model__lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'model__lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'model__bagging_seed': trial.suggest_int('bagging_seed', 0, 100),
        'model__class_weight': 'balanced'
    }

    # Define the pipeline with scaling, SMOTE, and the model
    pipeline = ImbPipeline([
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

        pipeline.fit(
            X_train, y_train,
            model__eval_set=[(X_valid, y_valid)],
            model__eval_metric='binary_logloss',
            model__callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
        )

        y_pred_prob = pipeline.predict_proba(X_valid)[:, 1]
        roc_auc = roc_auc_score(y_valid, y_pred_prob)
        roc_auc_scores.append(roc_auc)

    return np.mean(roc_auc_scores)

def optimize_lightgbm_hyperparameters(X, y, n_trials=50000):
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

# ===========================
# Processing Function
# ===========================

def process_delta_t(delta_t, PERC_DATA_USED, ask_file_name, bid_file_name, model_type, y_type, top_n_values, correlation_threshold, data_path, model_output_path, plotting=False):
    """
    Processes a single delta_t value: feature evaluation, selection, training, evaluation, and saving.
    Returns a dictionary of results.
    """
    print(f"\n==============================")
    print(f"Processing delta_t = {delta_t} hours or {round(delta_t / 6, 2)} days")
    print(f"==============================\n")
    
    # Initialize a local result dictionary
    local_result = {}
    
    try:
        # ===========================
        # Step 1: Feature Evaluation
        # ===========================
        print("Step 1: Feature Evaluation")
        feature_scores_file = feature_evaluation(
            ask_file_name=ask_file_name,
            bid_file_name=bid_file_name,
            PERC_DATA_USED=PERC_DATA_USED,
            delta_t=delta_t,
            model_type=model_type,
            y_types=y_type
        )

        # ===========================
        # Step 2: Feature Selection
        # ===========================
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

        # ===========================
        # Step 3: Data Loading and Preprocessing
        # ===========================
        print("\nStep 3: Data Loading and Preprocessing")
        data = correct_format(features_name)

        # We only train on the most important features
        X = data[included_features].values
        y = data['y'].values  # Assuming 'y' is the target variable

        # Correct format
        X = X.astype(np.float64)
        y = y.astype(np.int8)

        print(f"Selected Features Shape: {X.shape}")

        # ===========================
        # Step 4: Hyperparameter Tuning
        # ===========================
        print("\nStep 4: Hyperparameter Tuning with Optuna")
        best_params = optimize_lightgbm_hyperparameters(X, y, n_trials=50)

        # ===========================
        # Step 5: Train Final Model
        # ===========================
        print("\nStep 5: Training Final LightGBM Model with Nested Cross-Validation")
        tscv_outer = TimeSeriesSplit(n_splits=5)
        outer_fold = 0

        # Initialize lists to store metrics
        outer_metrics = {
            'accuracy': [],
            'roc_auc': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

        # Initialize SHAP explainer
        explainer = None
        shap_values_total = []

        for train_index, test_index in tscv_outer.split(X):
            outer_fold += 1
            print(f"\n--- Outer Fold {outer_fold} ---")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Inner CV for hyperparameter tuning is already done, use best_params
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('model', lgb.LGBMClassifier(**best_params, random_state=42, class_weight='balanced'))
            ])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Predict probabilities
            y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

            # Optimize decision threshold based on precision-recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optimal Threshold for Fold {outer_fold}: {optimal_threshold:.4f}")

            # Make predictions based on the optimal threshold
            y_pred = (y_pred_prob >= optimal_threshold).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_prob)
            precision_score_val = precision_score(y_test, y_pred, zero_division=0)
            recall_score_val = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Store metrics
            outer_metrics['accuracy'].append(acc)
            outer_metrics['roc_auc'].append(roc_auc)
            outer_metrics['precision'].append(precision_score_val)
            outer_metrics['recall'].append(recall_score_val)
            outer_metrics['f1_score'].append(f1)

            print(f"Fold {outer_fold} - Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}, Precision: {precision_score_val:.4f}, Recall: {recall_score_val:.4f}, F1-Score: {f1:.4f}")

            # SHAP Analysis
            if explainer is None:
                explainer = shap.TreeExplainer(pipeline.named_steps['model'])
            shap_values = explainer.shap_values(X_test)
            shap_values_total.append(shap_values)

        # Calculate average metrics
        avg_accuracy = np.mean(outer_metrics['accuracy'])
        avg_roc_auc = np.mean(outer_metrics['roc_auc'])
        avg_precision = np.mean(outer_metrics['precision'])
        avg_recall = np.mean(outer_metrics['recall'])
        avg_f1 = np.mean(outer_metrics['f1_score'])

        if plotting:
            print(f"\nAverage Metrics for delta_t={round(delta_t / 6, 2)} days ({delta_t} hours):")
            print(f"  Accuracy: {avg_accuracy:.4f}")
            print(f"  ROC AUC: {avg_roc_auc:.4f}")
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall: {avg_recall:.4f}")
            print(f"  F1-Score: {avg_f1:.4f}")

        # ===========================
        # Step 6: Save the Model
        # ===========================
        print("\nStep 6: Saving the Trained Model")
        model_filename = f'final_lightgbm_model_delta_t_{delta_t}.pkl'
        model_filepath = os.path.join(model_output_path, model_filename)
        joblib.dump(pipeline, model_filepath)
        print(f"Model saved at {model_filepath}")

        # ===========================
        # Step 7: Record Performance
        # ===========================
        print("\nStep 7: Feature Importance Analysis with SHAP")
        try:
            # Aggregate SHAP values
            shap_values_total = np.concatenate(shap_values_total, axis=0)
            feature_names = included_features

            # Summary plot
            shap.summary_plot(shap_values_total, X_test, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot for delta_t={round(delta_t / 6, 2)} days')
            plt.tight_layout()
            shap_plot_path = os.path.join(model_output_path, f'shap_summary_delta_t_{delta_t}.png')
            plt.savefig(shap_plot_path)
            plt.close()
            print(f"SHAP summary plot saved at {shap_plot_path}")
        except Exception as e:
            print(f"Error during SHAP analysis: {e}")

        # Compile local results
        local_result = {
            'delta_t': delta_t,
            'included_features': included_features,
            'average_accuracy': avg_accuracy,
            'average_roc_auc': avg_roc_auc,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_f1_score': avg_f1,
            'model_path': model_filepath,
            'error': None  # Indicate no error
        }

        return local_result

    except Exception as e:
        print(f"Error processing delta_t={delta_t}: {e}")
        # Ensure all keys are present even in case of error
        local_result = {
            'delta_t': delta_t,
            'included_features': None,
            'average_accuracy': None,
            'average_roc_auc': None,
            'average_precision': None,
            'average_recall': None,
            'average_f1_score': None,
            'model_path': None,
            'error': str(e)
        }
        return local_result




# ===========================
# Main Execution Workflow
# ===========================

# ===========================
# Configuration Parameters
# ===========================
# Data Parameters
PERC_DATA_USED = 0.8
# delta_t_values = [5, 10, 14, 20, 30, 35, 40, 50, 60, 70, 80, 100]  # in hours
delta_t_values = [12, 15, 16, 17, 18, 19, 25, 30, 32, 35, 36, 38, 40]  # in hours



y_type = ["binary_classifier"]
# File Parameters
ask_file_name = "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-31.10.2024.csv" 
# Feature Selection Parameters

#top_n_values = [5,6,7,8,9,10,11,12,13,14,15,16,18,20,25,30]
#top_n_values = [10,11,12,13,14,15,16,18]

top_n_values = [8,9,10,11,12,13,14,15,16,18,20,25,30]

correlation_threshold = 0.6
# Model Parameters
model_type = "lightgbm"  # Options: 'random_forest', 'xgboost', 'lightgbm'
# Output Paths
data_path = os.path.join(os.getcwd(), "data")
model_output_path = os.path.join(data_path, "models")  # Directory to save trained models
os.makedirs(model_output_path, exist_ok=True)
# Initialize a list to store results
results = []


for dt in delta_t_values:
    print(f"\n--- Starting processing for delta_t={dt} hours ---")
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
        plotting = False
    )
    results.append(result)

parallel_results = results


# ===========================
# Compile and Analyze Results
# ===========================
print("\n==============================")
print("Compilation of All Results")
print("==============================\n")
results_df = pd.DataFrame(parallel_results)
print(results_df)
if results_df.empty:
    print("No results to display.")
    exit()
# Optional: Filter out failed processes
failed_processes = results_df[results_df['error'].notnull()]
if not failed_processes.empty:
    print("\nFailed delta_t values and reasons:")
    print(failed_processes[['delta_t', 'error']])
    # Optionally, remove failed entries from results_df
    results_df = results_df[results_df['error'].isnull()]
# Proceed if there are successful results
if results_df.empty:
    print("All delta_t processes failed. Exiting.")
    exit()
# Identify the best delta_t based on ROC AUC
best_delta_t_row = results_df.loc[results_df['average_roc_auc'].idxmax()]
print(best_delta_t_row)
best_delta_t = best_delta_t_row['delta_t']
best_model_path = best_delta_t_row['model_path']
best_features = best_delta_t_row['included_features']
best_accuracy = best_delta_t_row['average_accuracy']
best_roc_auc = best_delta_t_row['average_roc_auc']
best_precision = best_delta_t_row['average_precision']
best_recall = best_delta_t_row['average_recall']
best_f1 = best_delta_t_row['average_f1_score']
print(f"\nOptimal delta_t based on ROC AUC: {round(best_delta_t / 6, 2)} days ({best_delta_t} hours)")
print(f"Best Model Path: {best_model_path}")
print(f"Selected Features: {best_features}")
print(f"Best Average Accuracy: {best_accuracy:.4f}")
print(f"Best Average ROC AUC: {best_roc_auc:.4f}")
print(f"Best Average Precision: {best_precision:.4f}")
print(f"Best Average Recall: {best_recall:.4f}")
print(f"Best Average F1-Score: {best_f1:.4f}")

# Optional: Visualize ROC AUC across different delta_t values
try:
    plt.figure(figsize=(12, 8))
    sns.barplot(x='delta_t', y='average_roc_auc', data=results_df, palette='viridis')
    plt.title('Average ROC AUC Across Different delta_t Values')
    plt.xlabel('delta_t (hours)')
    plt.ylabel('Average ROC AUC')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error during ROC AUC visualization: {e}")

