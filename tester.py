import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

# ------------------------------ Configuration ------------------------------ #

# Paths
MODEL_PATH = 'final_lightgbm_model_delta_t_36.pkl'  # Path to your saved model
#TEST_DATA_PATH = 'features_binary_classifierdelta_t701AAPL.USUSD_Candlestick_1_Hour_ASK_18.09.2024-30.10.2024.csv'  # Path to your test dataset
TEST_DATA_PATH = "features_binary_classifierdelta_t361AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv"
COLUMN_NAMES_PATH = 'column_names.csv'  # Path to column names (if separate)
REPORTS_DIR = 'reports'
PLOTS_DIR = os.path.join(REPORTS_DIR, 'plots')
METRICS_FILE = os.path.join(REPORTS_DIR, 'metrics.txt')

# Selected Features (same as trained model features)
SELECTED_FEATURES = ['score', 'Week_of_Year', 'VPT', 'Leading_Span_B', 'PCA_5', 'OBV', 'Signal_Line', 'Rolling_Skew', 'MACD_scaled ATR_scaled', 'Rolling_Kurt', 'Hist_Volatility']

# Visualization Settings
sns.set(style='whitegrid')

# ------------------------------ Utility Functions ------------------------------ #

def ensure_directories():
    """Ensure that the reports and plots directories exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Reports will be saved in '{REPORTS_DIR}' directory.")

def load_model(model_path):
    """Load the trained model from a .pkl file using joblib."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from '{model_path}'.")
        return model
    except Exception as e:
        print(f"Error loading the model with joblib: {e}")
        raise

def load_test_data(test_data_path):
    """Load test data from a CSV file."""
    try:
        data = pd.read_csv(test_data_path)
        print(f"Test data loaded successfully from '{test_data_path}'.")
        return data
    except Exception as e:
        print(f"Error loading test data: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the test data to select only the relevant features.
    """

    X = data[SELECTED_FEATURES].copy()
    y = data.iloc[:, -1]  # Assuming the last column is the target variable


    print("Data preprocessing completed.")
    return X, y
# ------------------------------ Evaluation and Plotting Functions ------------------------------ #

def evaluate_model(model, X, y, date):
    """Evaluate the model's performance on the test set."""
    print("Evaluating model performance...")
    y_pred = model.predict(X)
    
    # If it's a classifier with probability estimates
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:,1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)
    else:
        y_proba = y_pred  # Fallback
    
    metrics = {}
    metrics['ROC AUC'] = roc_auc_score(y, y_proba)
    metrics['Accuracy'] = accuracy_score(y, y_pred)
    metrics['Precision'] = precision_score(y, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y, y_pred, zero_division=0)
    metrics['F1 Score'] = f1_score(y, y_pred, zero_division=0)
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Classification Report
    class_report = classification_report(y, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(class_report)
    
    # Print actual vs. predicted values
    results_df = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred,
        'Probability': y_proba,
        'Date': date
    })
    print("\nActual vs. Predicted values:")

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    
    print(results_df)
    
    # Save metrics to a text file
    with open(METRICS_FILE, 'w') as f:
        f.write(f"Evaluation Metrics using {model}:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)
    print(f"Metrics saved to '{METRICS_FILE}'.")
    
    return y, y_pred, y_proba, metrics


def plot_roc_curve(y, y_proba):
    """Plot and save the ROC Curve."""
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y, y_proba):.4f})')
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC Curve saved to '{roc_path}'.")

def plot_confusion_matrix(y, y_pred):
    """Plot and save the Confusion Matrix."""
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to '{cm_path}'.")

# ------------------------------ Main Testing Workflow ------------------------------ #

def main():
    # Ensure necessary directories exist
    print(f"USING MODEL {MODEL_PATH}")
    ensure_directories()
    
    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Load the test data
    test_data = load_test_data(TEST_DATA_PATH)

    #print(int(0.7 * len(test_data)))
    test_data = test_data.iloc[int(0.7 * len(test_data)):]
    
    date = test_data['Datetime']



    # Preprocess the data (using only selected features)
    X, y = preprocess_data(test_data)
    
    # Evaluate the model
    y, y_pred, y_proba, metrics = evaluate_model(model, X, y, date)
    
    # Generate Plots
    plot_roc_curve(y, y_proba)
    plot_confusion_matrix(y, y_pred)
    print(f"USING MODEL {MODEL_PATH}")
    print("Testing completed successfully.")

if __name__ == '__main__':
    main()
