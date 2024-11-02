def feature_evaluation(ask_file_name, bid_file_name, PERC_DATA_USED, delta_t, model_type='random_forest', y_types=["binary_classifier"]):
    """
    Evaluate and compute feature importances using the specified model.

    Parameters:
    - ask_file_name: Path to the ASK CSV file
    - bid_file_name: Path to the BID CSV file
    - PERC_DATA_USED: Percentage of data to use (if applicable)
    - delta_t: Time delta for target variable
    - model_type: Model to use for feature importance ('random_forest', 'xgboost', 'lightgbm')
    - y_types: List of target types (default: ["binary_classifier"])

    Returns:
    - Path to the feature importances CSV file 
    """
    import pandas as pd
    import numpy as np
    import os
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegressionCV, LassoCV
    from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression, SelectKBest
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.feature_selection import RFE
    import xgboost as xgb
    import lightgbm as lgb
    import shap

    # Import custom modules (assuming they exist)
    from read_data import correct_format
    from feature_engineering import add_features

    # Set the path to your data directory
    path = os.path.join(os.getcwd(), "labeled_data")

    print("--Feature evaluation started!--")




    for y_type in y_types:
        output_file_name = f'feature_scores_{y_type}_delta_t{delta_t}_perc{PERC_DATA_USED}_{ask_file_name}'
        output_file_path = os.path.join(path, output_file_name)

        # If the file already exists, skip feature evaluation
        if not os.path.isfile(output_file_path):
            classifier = (y_type == "price_movements_classification") | (y_type == "binary_classifier")
            features_name = f'features_{y_type}delta_t{delta_t}{PERC_DATA_USED}{ask_file_name}'
            #features_bdelta_t120.8AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv'
            features_path = os.path.join(path, features_name)
            
            # Add features if they don't exist
            if not os.path.isfile(features_path):
                
                add_features(
                 ask_file_name =ask_file_name,
                 bid_file_name =bid_file_name, 
                 PRECENTAGE=PERC_DATA_USED, 
                 y_type=y_type, 
                 delta_t=delta_t)
                
            else:
                print(f"{features_name} already exists!")

            # Load and prepare data
            data = correct_format(features_name)
            X = data.iloc[:, 1:-1].values  # Assuming first column is 'Datetime' and last column is 'y'
            y = data.iloc[:, -1].values
            X = X.astype(np.float32)
            y = y.astype(np.int8) if classifier else y.astype(np.float32)

            # Define column names
            
            #dynamic
            column_names = data.columns[1:-1].tolist()
        

            # Initialize a DataFrame to store feature scores from different methods
            feature_scores_df = pd.DataFrame({'Feature': column_names})

            # Initialize models based on problem type
            if classifier:
                # Classification problem
                score_func = f_classif
                mutual_info_func = mutual_info_classif
                if model_type == 'random_forest':
                    model_instance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    importance_col = 'Random_Forest_Importance'
                elif model_type == 'xgboost':
                    model_instance = xgb.XGBClassifier(
                        random_state=42,
                        tree_method='gpu_hist',  # Use 'gpu_hist' if GPU is available
                        gpu_id=0
                    )
                    importance_col = 'XGBoost_Importance'
                elif model_type == 'lightgbm':
                    model_instance = lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=12,
                        random_state=42,
                        class_weight='balanced'  # Handle class imbalance
                    )
                    importance_col = 'LightGBM_Importance'
                else:
                    raise ValueError("Unsupported model_type. Choose from 'random_forest', 'xgboost', 'lightgbm'.")

                lasso_model = LogisticRegressionCV(
                    cv=5, penalty='l1', solver='saga', random_state=42, max_iter=5000, n_jobs=-1
                )
            else:
                # Regression problem
                score_func = f_regression
                mutual_info_func = mutual_info_regression
                if model_type == 'random_forest':
                    model_instance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    importance_col = 'Random_Forest_Importance'
                elif model_type == 'xgboost':
                    model_instance = xgb.XGBRegressor(
                        random_state=42,
                        tree_method='gpu_hist',
                        gpu_id=0
                    )
                    importance_col = 'XGBoost_Importance'
                elif model_type == 'lightgbm':
                    model_instance = lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=12,
                        random_state=42
                    )
                    importance_col = 'LightGBM_Importance'
                else:
                    raise ValueError("Unsupported model_type. Choose from 'random_forest', 'xgboost', 'lightgbm'.")

                lasso_model = LassoCV(cv=5, random_state=42, n_jobs=-1)

            # 1. Univariate Feature Selection (ANOVA F-test or f_regression)
            # Complexity: Low, runtime ~1 min
            print("1. Performing Univariate Feature Selection...")
            selector = SelectKBest(score_func=score_func, k='all')  # Select all features
            selector.fit(X, y)
            feature_scores_df['Univariate_Score'] = selector.scores_
            print('1. Univariate Feature Selection completed.')

            # 2. Mutual Information
            # Complexity: Low/Moderate, runtime ~3 min 
            print("2. Calculating Mutual Information...")
            mi_scores = mutual_info_func(X, y, n_jobs=-1)
            feature_scores_df['Mutual_Info_Score'] = mi_scores
            print('2. Mutual Information calculation completed.')

            # 3. Feature Importance from Specified Model
            # Complexity: Moderate, runtime ~5 min 
            print(f"3. Calculating Feature Importances using {model_type.replace('_', ' ').title()}...")
            model_instance.fit(X, y)
            importances = model_instance.feature_importances_
            feature_scores_df[importance_col] = importances
            print('3. Feature Importances calculation completed.')

            # 4. (Optional) Permutation Importance
            # Complexity: High, runtime: several hours
            # print("4. Calculating Permutation Importance...")
            # perm_importance = permutation_importance(
            #     model_instance, X, y, n_repeats=10, random_state=42, n_jobs=-1
            # )
            # feature_scores_df['Permutation_Importance'] = perm_importance.importances_mean
            # print('4. Permutation Importance calculation completed.')

            # 5. (Optional) Recursive Feature Elimination (RFE)
            # Complexity: Very High, runtime more hours to days
            # print("5. Performing Recursive Feature Elimination (RFE)...")
            # rfe_selector = RFE(estimator=model_instance, n_features_to_select=10, step=1)
            # rfe_selector.fit(X, y)
            # feature_scores_df['RFE_Ranking'] = rfe_selector.ranking_
            # print('5. RFE completed.')

            from sklearn.feature_selection import RFECV

            print("5. Performing Recursive Feature Elimination with Cross-Validation (RFECV)...")
            rfecv = RFECV(
                estimator=model_instance,
                step=2,  # Adjust step size based on dataset size
                cv=TimeSeriesSplit(n_splits=5),
                scoring='roc_auc',  # Optimize based on ROC AUC
                n_jobs=-1
            )
            rfecv.fit(X, y)

            # Use 'column_names' instead of 'selected_features'
            selected_features = [feature for feature, support in zip(column_names, rfecv.support_) if support]
            print(f"Optimal number of features: {rfecv.n_features_}")
            print("Selected Features after RFECV:")
            print(selected_features)

            feature_scores_df['RFECV_Ranking'] = rfecv.ranking_
            print('5. RFECV completed.')


            # 6. (Optional) Lasso (L1 Regularization)
            # Complexity: High, runtime ~20 min
            # print("6. Performing Lasso (L1 Regularization)...")
            # lasso_model.fit(X, y)
            # if y_type == "binary_classifier":
            #     lasso_coef = np.mean(np.abs(lasso_model.coef_), axis=0)  # Mean coefficient magnitude across folds
            # else:
            #     lasso_coef = np.abs(lasso_model.coef_)
            # feature_scores_df['Lasso_Coefficients'] = lasso_coef
            # print('6. Lasso completed.')

            # 7. (Optional) Correlation with Target Variable
            # Complexity: low, runtime: ~3 min
            # print("7. Calculating Correlation with Target Variable...")
            # if classifier:
            #     from scipy.stats import pointbiserialr
            #     correlations = Parallel(n_jobs=-1)(
            #         delayed(pointbiserialr)(X[:, i], y)[0] for i in range(X.shape[1])
            #     )
            #     feature_scores_df['Correlation_With_Target'] = correlations
            # else:
            #     correlations = pd.Series(X).corrwith(pd.Series(y)).values
            #     feature_scores_df['Correlation_With_Target'] = correlations
            # print('7. Correlation calculation completed.')

            # 4. Model-Specific Feature Importance (e.g., LightGBM)
            if model_type == 'lightgbm' and classifier:
                print("4. Calculating LightGBM-specific Feature Importances using SHAP...")
                # Initialize SHAP explainer
                explainer = shap.TreeExplainer(model_instance)
                shap_values = explainer.shap_values(X)
    
                # For binary classification, shap_values is a list with two arrays (for each class)
                # We take the mean absolute SHAP values for the positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap_importances = np.abs(shap_values).mean(axis=0)
    
                # Add SHAP importances to feature_scores_df
                feature_scores_df['SHAP_Importance'] = shap_importances
                print('4. LightGBM-specific Feature Importances (SHAP) calculation completed.')
    
            # 5. Correlation with Target Variable
            print("5. Calculating Correlation with Target Variable...")
            if classifier:
                from scipy.stats import pointbiserialr
                correlations = []
                for i in range(X.shape[1]):
                    corr, _ = pointbiserialr(X[:, i], y)
                    correlations.append(corr)
                feature_scores_df['Correlation_With_Target'] = correlations
            else:
                correlations = data.iloc[:, 1:-1].corrwith(data['y']).values
                feature_scores_df['Correlation_With_Target'] = correlations
            print('5. Correlation with Target Variable completed.')
    
            # Save the feature scores to a CSV file
            feature_scores_df.to_csv(output_file_path, index=False)
            print(f"Feature scores saved to {output_file_path}")

        else:
            print(f"File already exists at: {output_file_path}")
    
    return output_file_path


# ask_file_name = "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-26.10.2024.csv"
# bid_file_name = "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-26.10.2024.csv"

# PERC_DATA_USED = 1
# delta_t = 5
# y_type = ["binary_classifier"]
# # File Parameters
# ask_file_name = "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-26.10.2024.csv"
# bid_file_name = "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-26.10.2024.csv"
# model_type = "lightgbm"


# feature_scores_file = feature_evaluation(
#     ask_file_name=ask_file_name,
#     bid_file_name=bid_file_name,
#     PERC_DATA_USED=PERC_DATA_USED,
#     delta_t=delta_t,
#     model_type=model_type,
#     y_types=y_type
#         )

# print(feature_scores_file)