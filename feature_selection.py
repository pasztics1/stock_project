import pandas as pd
import numpy as np
import os

# Import necessary feature selection and evaluation tools
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# If using XGBoost for feature importance
import xgboost as xgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import lightgbm as lgb

# Import custom modules (assuming they exist)
from read_data import correct_format

# Import the updated feature_evaluation function
# Ensure that the updated feature_evaluation function is in the current namespace or imported appropriately
# from your_module import feature_evaluation

#THINGS TO FIX#
#The code at the if is redundant when handling the data.
#THINGS TO FIX#


def remove_highly_correlated_features(corr_matrix, features, feature_importances, importance_col, threshold=0.8):
    """
    Remove features that are highly correlated with each other.

    Parameters:
    - corr_matrix: DataFrame containing the correlation matrix
    - features: List of feature names
    - feature_importances: DataFrame containing feature names and their importance scores
    - importance_col: String indicating the column name for feature importances
    - threshold: Correlation threshold for removing features

    Returns:
    - List of features with high correlations removed
    """
    # Set to hold features to remove
    features_to_remove = set()

    # Iterate over the correlation matrix
    for i in range(len(features)):
        feature_i = features[i]
        if feature_i in features_to_remove:
            continue
        for j in range(i + 1, len(features)):
            feature_j = features[j]
            if feature_j in features_to_remove:
                continue
            correlation = corr_matrix.loc[feature_i, feature_j]
            if correlation >= threshold:
                # Compare importances
                importance_i = feature_importances.loc[feature_importances['Feature'] == feature_i, importance_col].values[0]
                importance_j = feature_importances.loc[feature_importances['Feature'] == feature_j, importance_col].values[0]
                # Remove the less important feature
                if importance_i >= importance_j:
                    features_to_remove.add(feature_j)
                else:
                    features_to_remove.add(feature_i)
    # Final list of features
    final_features = [feature for feature in features if feature not in features_to_remove]
    return final_features

def select_optimal_features(feature_scores_file, dataset_file, top_n_values, model="random_forest", correlation_threshold=0.8, plot=False):
    """
    Select the most optimal features based on feature importance and model performance.

    Parameters:
    - feature_scores_file: Path to the feature importance scores CSV file
    - dataset_file: Path to the dataset CSV file
    - top_n_values: List or range of 'top_n' values to evaluate
    - model: Model to use for feature importance ('random_forest' or 'light_gbm')
    - correlation_threshold: Correlation threshold for removing features
    - plot: Boolean to indicate whether to plot performance

    Returns:
    - optimal_features: List of the most optimal features
    """
    import os
    path = os.getcwd()

    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    # Read the feature importance scores
    feature_scores_path = os.path.join(path,feature_scores_file)
    feature_scores_df = pd.read_csv(feature_scores_path)


    #Read the dataset
    dataset_path = os.path.join(path,dataset_file)

    print(dataset_file)

    data = correct_format(dataset_file)

    #we only train on the most important features

    y = data.iloc[:, -1].values
    y = y.astype(np.int8) #!!!!IF NOT CLASSIFIER, IT CAN CAUSE PROBLEMS!!!!


    
    performance_metrics = []

    if model == "random_forest":
        importance_col = 'Random_Forest_Importance'
        for top_n in top_n_values:
            print(f"\nEvaluating top_n = {top_n}")
            # Sort the features by 'Random_Forest_Importance' in descending order
            top_features_rf = feature_scores_df.sort_values(by=importance_col, ascending=False)

            # Get the list of top features
            top_features_list = top_features_rf['Feature'].head(top_n).tolist()
            print(f"Top {top_n} Features Based on Random Forest Importance:")
            print(top_features_list)

            # Extract the data for the top features
            top_features_data = data[top_features_list].iloc[:,:].values
            top_features_data_df = pd.DataFrame(top_features_data, columns=top_features_list)

            # Compute the correlation matrix
            corr_matrix = top_features_data_df.corr().abs()
            corr_matrix.fillna(0, inplace=True)  # Handle any NaN values

            # Remove highly correlated features
            selected_features = remove_highly_correlated_features(
                corr_matrix, top_features_list, top_features_rf, importance_col, threshold=correlation_threshold
            )

            # Number of features removed
            n_removed = len(top_features_list) - len(selected_features)
            print(f"Features after removing those with correlation >= {correlation_threshold}:")
            print(selected_features)
            print(f"Number of features removed due to high correlation: {n_removed}")

            # Prepare data with selected features
            X = data[selected_features].iloc[:,:].values

            # Convert data types if necessary
            X = X.astype(np.float32)
            y = y.astype(np.int8)

            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)

            # Lists to store metrics
            fold_accuracies = []
            fold_cv_accuracies = []

            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Train the Random Forest model
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'  # Handle class imbalance
                )
                rf_model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

                # Cross-validation accuracy
                cv_scores = cross_val_score(
                    rf_model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1
                )
                mean_cv_score = cv_scores.mean()
                fold_cv_accuracies.append(mean_cv_score)
                print(f"Fold {fold + 1} Cross-Validation Accuracy: {mean_cv_score:.4f}")

            # Average metrics across folds
            avg_accuracy = np.mean(fold_accuracies)
            avg_cv_accuracy = np.mean(fold_cv_accuracies)

            print(f"Average Accuracy with {len(selected_features)} features: {avg_accuracy:.4f}")
            print(f"Average Cross-Validated Accuracy: {avg_cv_accuracy:.4f}")

            # Record performance metrics
            performance_metrics.append({
                'n_features_selected': len(selected_features),
                'top_n': top_n,
                'accuracy': avg_accuracy,
                'cv_accuracy': avg_cv_accuracy,
                'selected_features': selected_features
            })

    elif model == "lightgbm":
        importance_col = 'LightGBM_Importance'
        for top_n in top_n_values:
            print(f"\nEvaluating top_n = {top_n}")
            # Sort the features by 'LightGBM_Importance' in descending order
            top_features_lgbm = feature_scores_df.sort_values(by=importance_col, ascending=False)

            # Get the list of top features
            top_features_list = top_features_lgbm['Feature'].head(top_n).tolist()
            print(f"Top {top_n} Features Based on LightGBM Importance:")
            print(top_features_list)

            # Extract the data for the top features
            top_features_data = data[top_features_list].iloc[:,:].values
            top_features_data_df = pd.DataFrame(top_features_data, columns=top_features_list)


            # Compute the correlation matrix
            corr_matrix = top_features_data_df.corr().abs()
            corr_matrix.fillna(0, inplace=True)  # Handle any NaN values

            # Remove highly correlated features
            selected_features = remove_highly_correlated_features(
                corr_matrix, top_features_list, top_features_lgbm, importance_col, threshold=correlation_threshold
            )

            print(f'\n{selected_features}\n')

            # Number of features removed
            n_removed = len(top_features_list) - len(selected_features)
            print(f"Features after removing those with correlation >= {correlation_threshold}:")
            print(selected_features)
            print(f"Number of features removed due to high correlation: {n_removed}")

            # Prepare data with selected features
            X = data[selected_features].iloc[:,:].values

            # Convert data types if necessary
            X = X.astype(np.float32)
            y = y.astype(np.int8)

            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)

            # Lists to store metrics
            fold_accuracies = []
            fold_cv_accuracies = []
            fold_roc_aucs = []

            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train the LightGBM model
                lgbm_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=12,
                    random_state=42,
                    class_weight='balanced'  # Handle class imbalance
                )

                # Train the model with the fit() method
                lgbm_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],  # Validation data for early stopping
                    eval_metric='binary_logloss',  # Metric to evaluate
                    callbacks=[early_stopping(stopping_rounds=10), log_evaluation(0)]  # Use callbacks for early stopping and logging
                )



                # Predict and evaluate
                y_pred = lgbm_model.predict(X_test)
                y_pred_prob = lgbm_model.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_prob)
                fold_accuracies.append(accuracy)
                fold_roc_aucs.append(roc_auc)
                print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
                print(f"Fold {fold + 1} ROC AUC: {roc_auc:.4f}")

                # Cross-validation accuracy
                cv_scores = cross_val_score(
                    lgbm_model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1
                )
                mean_cv_score = cv_scores.mean()
                fold_cv_accuracies.append(mean_cv_score)
                print(f"Fold {fold + 1} Cross-Validation Accuracy: {mean_cv_score:.4f}")

            # Average metrics across folds
            avg_accuracy = np.mean(fold_accuracies)
            avg_cv_accuracy = np.mean(fold_cv_accuracies)
            avg_roc_auc = np.mean(fold_roc_aucs)

            print(f"Average Accuracy with {len(selected_features)} features: {avg_accuracy:.4f}")
            print(f"Average Cross-Validated Accuracy: {avg_cv_accuracy:.4f}")
            print(f"Average ROC AUC: {avg_roc_auc:.4f}")

            # Record performance metrics
            performance_metrics.append({
                'n_features_selected': len(selected_features),
                'top_n': top_n,
                'accuracy': avg_accuracy,
                'cv_accuracy': avg_cv_accuracy,
                'roc_auc': avg_roc_auc,
                'selected_features': selected_features
            })

    # Find the top_n with the highest cross-validation accuracy or ROC AUC
    performance_df = pd.DataFrame(performance_metrics)

    if not performance_df.empty:
        if model == "random_forest":
            optimal_row = performance_df.loc[performance_df['cv_accuracy'].idxmax()]
            optimal_features = optimal_row['selected_features']
            optimal_n_features = optimal_row['n_features_selected']
            optimal_accuracy = optimal_row['cv_accuracy']
            print(f"\nOptimal number of features: {optimal_n_features}")
            print(f"Cross-Validated Accuracy: {optimal_accuracy:.4f}")
            print("Optimal Features:")
            for i, feature in enumerate(optimal_features, start=1):
                print(f"{i}. {  feature}")

        elif model == "lightgbm":
            optimal_row = performance_df.loc[performance_df['roc_auc'].idxmax()]
            optimal_features = optimal_row['selected_features']
            optimal_n_features = optimal_row['n_features_selected']
            optimal_roc_auc = optimal_row['roc_auc']
            print(f"\nOptimal number of features: {optimal_n_features}")
            print(f"Cross-Validated ROC AUC: {optimal_roc_auc:.4f}")
            print("Optimal Features:")
            for i, feature in enumerate(optimal_features, start=1):
                print(f"{i}. {feature}")

        if plot:
            # Plot the performance
            plt.figure(figsize=(10, 6))
            if model == "random_forest":
                plt.plot(performance_df['n_features_selected'], performance_df['cv_accuracy'], marker='o', label='CV Accuracy')
                plt.ylabel('Cross-Validated Accuracy')
            elif model == "lightgbm":
                plt.plot(performance_df['n_features_selected'], performance_df['roc_auc'], marker='o', label='ROC AUC')
                plt.ylabel('Cross-Validated ROC AUC')
            plt.xlabel('Number of Features Selected')
            plt.title(f'Model Performance vs. Number of Features ({model})')
            plt.grid(True)
            plt.legend()
            plt.show()
    else:
        print("No performance metrics recorded. Please check your model implementation.")
    return optimal_features




# PERC_DATA_USED = 1
# delta_t = 5
# use_xgboost = False
# y_type = "binary_classifier"

# ask_file_name = "AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv"
# bid_file_name = "AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv"
# features_name = f'features_{y_type}delta_t{delta_t}{PERC_DATA_USED}{ask_file_name}'

# top_n_values = range(15, 31, 5)

# feature_scores_file = feature_evaluation(ask_file_name,bid_file_name,PERC_DATA_USED,delta_t,use_xgboost)
# optimal_features = select_optimal_features(feature_scores_file, features_name, top_n_values, correlation_threshold=0.8, plot=True)

# print('Optimal features:\n',optimal_features)





# top_n_values = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20]
# delta_t = 5
# use_xgboost = False
# y_type = "binary_classifier"
# feature_scores_file = r'C:\Users\CsP\Desktop\stock_project-master\data\feature_scores_binary_classifier_delta_t14_perc0.8_AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv'
# features_name = 'features_binary_classifierdelta_t170.8AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv'
# model_type = "lightgbm"
# correlation_threshold = 0.7

# included_features = select_optimal_features(
#         feature_scores_file=feature_scores_file,
#         dataset_file=features_name,
#         top_n_values=top_n_values,
#         model=model_type,
#         correlation_threshold=correlation_threshold,
#         plot=True
#         )

# print(included_features)