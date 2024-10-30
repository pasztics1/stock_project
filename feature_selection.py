import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Import custom functions
from read_data import correct_format
from feature_evaluation import feature_evaluation

def remove_highly_correlated_features(corr_matrix, features, feature_importances, threshold=0.8):
    """
    Remove features that are highly correlated with each other.

    Parameters:
    - corr_matrix: DataFrame containing the correlation matrix
    - features: List of feature names
    - feature_importances: DataFrame containing feature names and their importance scores
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
                importance_i = feature_importances.loc[feature_importances['Feature'] == feature_i, 'Random_Forest_Importance'].values[0]
                importance_j = feature_importances.loc[feature_importances['Feature'] == feature_j, 'Random_Forest_Importance'].values[0]
                # Remove the less important feature
                if importance_i >= importance_j:
                    features_to_remove.add(feature_j)
                else:
                    features_to_remove.add(feature_i)
    # Final list of features
    final_features = [feature for feature in features if feature not in features_to_remove]
    return final_features

def select_optimal_features(feature_scores_file, dataset_file, top_n_values, model="random_forest",correlation_threshold=0.8, plot=False):
    """
    Select the most optimal features based on feature importance and model performance.

    Parameters:
    - feature_scores_file: Path to the feature importance scores CSV file
    - dataset_file: Path to the dataset CSV file
    - top_n_values: List or range of 'top_n' values to evaluate
    - correlation_threshold: Correlation threshold for removing features

    Returns:
    - optimal_features: List of the most optimal features
    """
    # Set the path to your data directory
    path = os.path.join(os.getcwd(), "data")

    # Paths to the files
    feature_scores_path = os.path.join(path, feature_scores_file)
    dataset_path = os.path.join(path, dataset_file)

    # Read the feature importance scores
    feature_scores_df = pd.read_csv(feature_scores_path)

    # Load the dataset
    data = correct_format(dataset_file)
    y = data['y']
    X_all = data.drop('y', axis=1)

    performance_metrics = []
    if model=="random_forest":
        for top_n in top_n_values:
            print(f"\nEvaluating top_n = {top_n}")
            # Sort the features by 'Random_Forest_Importance' in descending order
            top_features_rf = feature_scores_df.sort_values(by='Random_Forest_Importance', ascending=False)

            # Get the list of top features
            top_features_list = top_features_rf['Feature'].head(top_n).tolist()
            print(f"Top {top_n} Features Based on Random Forest Importance:")
            print(top_features_list)

            # Extract the data for the top features
            top_features_data = X_all[top_features_list]

            # Compute the correlation matrix
            corr_matrix = top_features_data.corr().abs()
            corr_matrix.fillna(0, inplace=True)  # Handle any NaN values

            # Remove highly correlated features
            selected_features = remove_highly_correlated_features(corr_matrix, top_features_list, top_features_rf, threshold=correlation_threshold)

            # Number of features removed
            n_removed = len(top_features_list) - len(selected_features)
            print(f"Features after removing those with correlation >= {correlation_threshold}:")
            print(selected_features)
            print(f"Number of features removed due to high correlation: {n_removed}")

            # Prepare data with selected features
            X = X_all[selected_features]

            # Convert data types if necessary
            X = X.astype(np.float32)
            y = y.astype(np.int8)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train the Random Forest model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy with {len(selected_features)} features: {accuracy:.4f}")

            # Cross-validation score
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring='accuracy', n_jobs=-1
            )
            mean_cv_score = cv_scores.mean()
            print(f"Cross-Validation Accuracy with {len(selected_features)} features: {mean_cv_score:.4f}")

            # Record performance metrics
            performance_metrics.append({
                'n_features_selected': len(selected_features),
                'top_n': top_n,
                'accuracy': accuracy,
                'cv_accuracy': mean_cv_score,
                'selected_features': selected_features
            })

        # Find the top_n with the highest cross-validation accuracy
        performance_df = pd.DataFrame(performance_metrics)
        optimal_row = performance_df.loc[performance_df['cv_accuracy'].idxmax()]
        optimal_features = optimal_row['selected_features']
        optimal_n_features = optimal_row['n_features_selected']
        optimal_accuracy = optimal_row['cv_accuracy']

        print(f"\nOptimal number of features: {optimal_n_features}")
        print(f"Cross-Validated Accuracy: {optimal_accuracy:.4f}")
        print("Optimal Features:")
        for i, feature in enumerate(optimal_features, start=1):
            print(f"{i}. {feature}")

        if plot:
            # Plot the performance
            plt.figure(figsize=(10, 6))
            plt.plot(performance_df['n_features_selected'], performance_df['cv_accuracy'], marker='o')
            plt.xlabel('Number of Features Selected')
            plt.ylabel('Cross-Validated Accuracy')
            plt.title('Model Performance vs. Number of Features')
            plt.grid(True)
            plt.show()

    elif model == "light_gbm":
        for top_n in top_n_values:
            print(f"\nEvaluating top_n = {top_n}")
            # Sort the features by 'LightGBM_Importance' in descending order
            top_features_lgbm = feature_scores_df.sort_values(by='Importance', ascending=False)

            # Get the list of top features
            top_features_list = top_features_lgbm['Feature'].head(top_n).tolist()
            print(f"Top {top_n} Features Based on LightGBM Importance:")
            print(top_features_list)

            # Extract the data for the top features
            top_features_data = X_all[top_features_list]

            # Compute the correlation matrix
            corr_matrix = top_features_data.corr().abs()
            corr_matrix.fillna(0, inplace=True)  # Handle any NaN values

            # Remove highly correlated features
            selected_features = remove_highly_correlated_features(corr_matrix, top_features_list, top_features_lgbm, threshold=correlation_threshold)

            # Number of features removed
            n_removed = len(top_features_list) - len(selected_features)
            print(f"Features after removing those with correlation >= {correlation_threshold}:")
            print(selected_features)
            print(f"Number of features removed due to high correlation: {n_removed}")

            # Prepare data with selected features
            X = X_all[selected_features]

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
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Train the LightGBM model
                lgbm_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=12,
                    random_state=42,
                    class_weight='balanced'  # Handle class imbalance
                )
                lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                               eval_metric='binary_logloss', early_stopping_rounds=10, verbose=False)

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

    # Find the top_n with the highest cross-validation accuracy
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
                print(f"{i}. {feature}")

        elif model == "light_gbm":
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
            elif model == "light_gbm":
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