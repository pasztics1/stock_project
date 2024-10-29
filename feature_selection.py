import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Import necessary modules for model training and evaluation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score
)

# Import custom functions
from read_data import correct_format


# Function to remove highly correlated features
def remove_highly_correlated_features(corr_matrix, features, threshold=0.8):
    """
    Remove features that are highly correlated with each other.

    Parameters:
    - corr_matrix: DataFrame containing the correlation matrix
    - features: List of feature names
    - threshold: Correlation threshold for removing features

    Returns:
    - List of features with high correlations removed
    """
    # Set to hold features to remove
    features_to_remove = set()
    features_to_keep = set(features)
    
    # Iterate over the correlation matrix
    for i in range(len(features)):
        feature_i = features[i]
        if feature_i in features_to_remove:
            continue
        for j in range(i + 1, len(features)):
            feature_j = features[j]
            if feature_j in features_to_remove:
                continue
            correlation = corr_matrix.iloc[i, j]
            if correlation >= threshold:
                # Compare importances
                importance_i = top_features_rf.loc[top_features_rf['Feature'] == feature_i, 'Random_Forest_Importance'].values[0]
                importance_j = top_features_rf.loc[top_features_rf['Feature'] == feature_j, 'Random_Forest_Importance'].values[0]
                # Remove the less important feature
                if importance_i >= importance_j:
                    features_to_remove.add(feature_j)
                else:
                    features_to_remove.add(feature_i)
    
    # Final list of features
    final_features = [feature for feature in features if feature not in features_to_remove]
    return final_features


# File names (replace these with your actual file names)
feature_scores_file = 'feature_scores_binary_classifier_delta_t5_perc1_AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv'
dataset_file = 'features_binary_classifierdelta_t51AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv'

# Select the top n features based on Random Forest importance
top_n = 20  


def relevant_features(feature_scores_file,dataset_file, top_n):
    # Set the path to your data directory
    path = os.path.join(os.getcwd(), "data")


    # Paths to the files
    feature_scores_path = os.path.join(path, feature_scores_file)
    dataset_path = os.path.join(path, dataset_file)

    # Read the feature importance scores
    feature_scores_df = pd.read_csv(feature_scores_path)

    # Sort the features by 'Random_Forest_Importance' in descending order
    top_features_rf = feature_scores_df.sort_values(by='Random_Forest_Importance', ascending=False)

    # Get the list of top features
    top_features_list = top_features_rf['Feature'].head(top_n).tolist()
    print("Top Features Based on Random Forest Importance:")
    print(top_features_list)

    # Load the dataset
    X = correct_format(dataset_file)

    # Extract the data for the top features
    top_features_data = X[top_features_list]

    # Compute the correlation matrix
    corr_matrix = top_features_data.corr().abs()

    # Visualize the correlation matrix (optional)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Top Features')
    plt.show()


    # Apply the function to remove highly correlated features
    selected_features = remove_highly_correlated_features(corr_matrix, top_features_list, threshold=0.8)

    # Print the results
    print(f"\nFeatures after removing those with correlation >= 0.8:")
    print(selected_features)

    # Number of features removed
    n_removed = len(top_features_list) - len(selected_features)
    print(f"\nNumber of features removed due to high correlation: {n_removed}")

    return selected_features
