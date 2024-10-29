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



# Import custom modules (assuming they exist)
from main import RF, RF_boosted  # Your custom Random Forest functions
from feature_engineering import add_features
from read_data import correct_format
#hyperparams
PERC_DATA_USED = 1
delta_t = 5
use_xgboost = False

# initializing file names for i/o
ask_file_name = "AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv"

# ask_file_name = "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-26.10.2024.csv"
# bid_file_name = "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-26.10.2024.csv"


y_types = ["binary_classifier"]#,"percentage_change","price_movements_classification"]
#adding feature engineering to our data if it doesn't exist


def feature_evaluation(ask_file_name,bid_file_name,PERC_DATA_USED,delta_t,use_xgboost,y_types=["binary_classifier"]):
    # Set the path to your data directory
    path = os.path.join(os.getcwd(), "data")


    for y_type in y_types:
        classifier =  (y_type == "price_movements_classification") | (y_type == "binary_classifier")
        features_name = f'features_{y_type}delta_t{delta_t}{PERC_DATA_USED}{ask_file_name}'
        features_path = os.path.join(path,features_name)

        if not os.path.isfile(features_path):
            add_features(ask_file_name,bid_file_name,PERC_DATA_USED, y_type, delta_t)
        else:
            print(f"{features_name} already exists!")

        data = correct_format(features_name)
        X = data.iloc[:, 1:-1].values #first row's not included, since it's date
        y = data.iloc[:, -1].values

        X = X.astype(np.float32)
        y = y.astype(np.int8) if classifier else y.astype(np.float32)

        # Get the column names (excluding 'Datetime' and 'y')
        column_names = [
        'Open_BID', 'High_BID', 'Low_BID', 'Close_BID', 'Volume_BID',
        'Open_ASK', 'High_ASK', 'Low_ASK', 'Close_ASK', 'Volume_ASK', 'Mid_Price',
        'Spread', 'Volume', 'Open_Mid', 'High_Mid', 'Low_Mid', 'Close_Mid',
        'Unix_Diff_Until_Earnings', 'Month', 'Day_Type', 'Week_of_Year', 'Quarter',
        'Hour', 'Is_Open_Hour', 'Is_Close_Hour', 'Day_of_Week', 'Is_Lunch_Time',
        'Time_Since_Open', 'score', 'rating', 'SMA_3', 'SMA_6', 'SMA_12', 'SMA_5',
        'SMA_10', 'SMA_20', 'EMA_3', 'EMA_6', 'EMA_12', 'EMA_26', 'MACD',
        'Signal_Line', '%K', '%D', 'CCI', 'Return_1H', 'Return_3H', 'Return_6H',
        'RSI_long', 'RSI_short', 'Middle_Band', 'Upper_Band', 'Lower_Band',
        'High_Low', 'High_Close', 'Low_Close', 'True_Range', 'ATR', 'Volume_SMA_3',
        'Volume_SMA_6', 'OBV', 'MF_Multiplier', 'MF_Volume', 'CMF', 'Volatility',
        'Return', 'Volume_change', 'Lag_Mid_Price_1', 'Lag_Mid_Price_2',
        'Lag_Mid_Price_3', 'Lag_Mid_Price_4', 'Lag_Return_1', 'Lag_Return_2',
        'Lag_Return_3', 'Lag_Return_4', 'Return_1', 'Return_5', 'Return_10',
        'Rolling_Mean_5', 'Rolling_Std_5', 'Lagged_Volatility', 'Hammer',
        'Rolling_Skew', 'Rolling_Kurt', 'Price_Z_Score', 'Conversion_Line',
        'Base_Line', 'Leading_Span_A', 'Leading_Span_B', 'Williams_%R',
        'High_Low_Ratio', 'Close_Open_Ratio', 'Order_Imbalance', 'VPT', 'A/D_Line',
        'Support_10', 'Resistance_10', 'Support_Close_10', 'Resistance_Close_10',
        'Support_20', 'Resistance_20', 'Support_Close_20', 'Resistance_Close_20',
        'SMA5_Volume', 'RSI_Volatility'
    ]


        # Prepare a DataFrame to store feature scores from different methods
        feature_scores_df = pd.DataFrame({'Feature': column_names})

        # Initialize the models and functions based on the problem type
        if classifier:
            # Classification problem
            score_func = f_classif
            mutual_info_func = mutual_info_classif
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            lasso_model = LogisticRegressionCV(
                cv=5, penalty='l1', solver='saga', random_state=42, max_iter=5000, n_jobs=-1)
        else:
            # Regression problem
            score_func = f_regression
            mutual_info_func = mutual_info_regression
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            lasso_model = LassoCV(cv=5, random_state=42, n_jobs=-1)
        
        # 1. Univariate Feature Selection (ANOVA F-test or f_regression)
        # Complexity: Low, runtime ~1 min
        selector = SelectKBest(score_func=score_func, k='all')  # Select all features
        selector.fit(X, y)
        feature_scores_df['Univariate_Score'] = selector.scores_
        
        print('1. ready')

        # 2. Mutual Information
        # Complexity: Low/Moderate, runtime ~3 min
        mi_scores = mutual_info_func(X, y, n_jobs=-1)
        feature_scores_df['Mutual_Info_Score'] = mi_scores
        
        print('2. ready')



        # 3. Feature Importance from Random Forest
        # Complexity: Moderate, runtime ~5 min
        model.fit(X, y)
        importances = model.feature_importances_
        feature_scores_df['Random_Forest_Importance'] = importances
        
        print('3. ready')

        # 4. Permutation Importance
        # Complexity: High, runtime: several hours
        # perm_importance = permutation_importance(
        # model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        #)
        # feature_scores_df['Permutation_Importance'] = perm_importance.importances_mean
        
        # print('4. ready')

        # 5. Recursive Feature Elimination (RFE)
        # Complexity: Very High, runtime more hours to days
        # rfe_selector = RFE(estimator=model, n_features_to_select=10, step=1)
        # rfe_selector.fit(X, y)
        # feature_scores_df['RFE_Ranking'] = rfe_selector.ranking_
        
        # print('5. ready')

        # 6. Lasso (L1 Regularization)
        # Complexity: High, runtime ~20 min
        lasso_model.fit(X, y)
        if y_type == "binary_classifier":
            lasso_coef = np.mean(np.abs(lasso_model.coef_), axis=0)  # Mean coefficient magnitude across folds
        else:
            lasso_coef = np.abs(lasso_model.coef_)
        feature_scores_df['Lasso_Coefficients'] = lasso_coef
        
        # print('6. ready')

        # 7. Correlation with Target Variable
        # Complexity: low, runtime: ~3 min
        if classifier:
            #For classification, compute point-biserial correlation
            from scipy.stats import pointbiserialr
            correlations = Parallel(n_jobs=-1)(
                delayed(pointbiserialr)(X[col], y)[0] for col in X.columns
            )
            feature_scores_df['Correlation_With_Target'] = correlations
        else:
            # For regression, use Pearson correlation
            correlations = X.corrwith(y)
            feature_scores_df['Correlation_With_Target'] = correlations.values
        
        print('7. ready')

        # 8. XGBoost Feature Importance 
        # Complexity: Moderate/High, runtime ~15 min
        if use_xgboost:
            if classifier:
                xgb_model = xgb.XGBClassifier(
                    random_state=42,
                    tree_method='gpu_hist',
                    gpu_id=0
                )
            else:
                xgb_model = xgb.XGBRegressor(
                    random_state=42,
                    tree_method='gpu_hist',
                    gpu_id=0
                )

            xgb_model.fit(X, y)
            xgb_importances = xgb_model.feature_importances_
            feature_scores_df['XGBoost_Importance'] = xgb_importances
            print('8. ready')


        
        # Save the feature scores to a CSV file
        output_file_name = f'feature_scores_{y_type}_delta_t{delta_t}_perc{PERC_DATA_USED}_{ask_file_name}'
        output_file_path = os.path.join(path, output_file_name)
        feature_scores_df.to_csv(output_file_path, index=False)
        
        print(f"Feature scores saved to {output_file_path}")
