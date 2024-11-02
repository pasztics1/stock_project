# # --------------------------- Function for joblib --------------------------- #

def store_feature_names(X, y=None):
    # Save feature names if the dataframe has columns
    if hasattr(X, 'columns'):
        store_feature_names.feature_names = X.columns.tolist()
    return X


# from hyperparam_tuning_optuna import store_feature_names

# ------------------------------ Configuration ------------------------------ #

#Hyperparams
TEST_DATA_USED = 0.2
y_type = "binary_classifier"
delta_t_values = [12,15,16,17,18,19,25,30,32,35,36,38,40]
threshold_optimizer = "pr_curve" #can be "iterative" & "pr_curve"
save_metrics = 'write' #can be write and add, if write it creates a new txt everytime, if add, it adds to the txt.

N_TOP_MODELS = 5 #number of top models saved 

# File Parameters
ask_file_name = "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv"
bid_file_name = "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-31.10.2024.csv" 


def evaluate_models(ask_file_name,bid_file_name,TEST_DATA_USED,delta_t_values,threshold_optimizer="pr_curve",save_metrics='write',N_TOP_MODELS=3):

    # ------------------------------ Imports ------------------------------ #

    import os
    import shutil
    
    import joblib
    
    import pandas as pd
    import numpy as np


    import matplotlib#!
    matplotlib.use('Agg')#!
    import matplotlib.pyplot as plt
    import seaborn as sns

    from datetime import datetime
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        classification_report,
        precision_recall_curve
    )

    from feature_engineering import add_features


    # ------------------------------ Utility Functions ------------------------------ #

    def ensure_directories():
        """Ensure that the reports and plots directories exist."""
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(BEST_MODELS_DIR, exist_ok=True)
        if ~((os.path.exists(SAVED_MODELS_DIR))&(os.path.exists(DATA_DIR))):
            print(f'!!!MODELS OR DATA NOT SAVED IN THE RIGHT FOLDER!!!')
        else:
            print(f'Labeled data in {DATA_DIR}')
            print(f'Saved models in {SAVED_MODELS_DIR}')
            print(f'Best models will be saved in {BEST_MODELS_DIR}')
            print(f"Reports will be saved in '{REPORTS_DIR}'")


    def load_model(model_path):
        """Load the trained model from a .pkl file using joblib."""
        # Retrieve feature names from the pipeline
        model = joblib.load(model_path)
        feature_names = model.named_steps['model'].original_feature_names
        print(f"Model loaded successfully from '{model_path}'.")
        print(f"Feature Names: {feature_names}")
        return model, feature_names



    def load_test_data(test_data_path):
        """Load test data from a CSV file."""
        try:
            data = pd.read_csv(test_data_path)
            print(f"Test data loaded successfully from '{test_data_path}'.")
            return data
        except Exception as e:
            print(f"Error loading test data: {e}")
            raise

    def preprocess_data(data,SELECTED_FEATURES):
        """
        Preprocess the test data to select only the relevant features.
        """
        X = data[SELECTED_FEATURES].copy()
        y = data.iloc[:, -1]  # the last column is the target variable
        print("Data preprocessing completed.")
        return X, y

    # ------------------------------ Evaluation and Plotting Functions ------------------------------ #

    def evaluate_model_with_threshold(model, X, y, threshold, feature_names, abs=False):
        """
        Evaluate the model's performance on the test set using a specific threshold.
        Returns y_high_conf, y_pred_high_conf, y_proba_high_conf, and the computed metrics.
        """
        # Obtain predicted probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X)
        else:
            y_proba = model.predict(X)  # Fallback if no probabilities
        
        if abs:
            # Apply the decision threshold to include both high-confidence positives and negatives
            high_confidence_indices = abs(y_proba - 0.5) >= (threshold)
        else:
            high_confidence_indices = y_proba >= (threshold)        
        
        
        if len(high_confidence_indices)>0:
            
            # Filter predictions based on the threshold
            y_high_conf = y[high_confidence_indices]
            y_pred_high_conf = (y_proba[high_confidence_indices] >= 0.5).astype(int)
            y_proba_high_conf = y_proba[high_confidence_indices]

            discarded_percentage = 100*(1-len(y_high_conf)/len(y))

        # Calculate metrics
        


        if ~(((y_high_conf == 1).any().any()) & ((y_high_conf == 1).any().any())): #Only save these if there are predictions with higher prob. than the curr. threshold & there are two classes
            roc_score = 0
        else:
            roc_score = roc_auc_score(y_high_conf, y_proba_high_conf)

        metrics = {
            'Threshold': threshold,  # Ensure this key is present and correctly capitalized
            'ROC AUC': roc_score,
            'Accuracy': accuracy_score(y_high_conf, y_pred_high_conf),
            'Precision': precision_score(y_high_conf, y_pred_high_conf, zero_division=0),
            'Recall': recall_score(y_high_conf, y_pred_high_conf, zero_division=0),
            'F1 Score': f1_score(y_high_conf, y_pred_high_conf, zero_division=0),
            'Guessed 0': int(sum(y_high_conf==0)),
            'Guessed 1' : int(sum(y_high_conf==1)),
            'Total' : len(y),
            'Avg_guess' : np.mean(y_proba),
            'Discarded' : int(len(y)-len(y_high_conf)),
            'Discarded Percentage': discarded_percentage,
            'Current time':datetime.now()
        }
        return y_high_conf, y_pred_high_conf, y_proba_high_conf, metrics


    def find_optimal_threshold(model, X, y, thresholds=np.arange(0.0, 0.5, 0.01)):
        """
        Iterate over a range of thresholds to find the optimal one based on desired metrics.
        Returns a DataFrame containing metrics for each threshold.
        """
        metrics_list = []
        for thresh in thresholds:
            y_high_conf, y_pred_high_conf, y_proba_high_conf, metrics = evaluate_model_with_threshold(model, X, y, thresh, abs=True)

            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df


    def find_optimal_threshold_pr_curve(model, X, y_true, feature_names): #y_true = y
        """
        Find the optimal threshold using the precision-recall curve to maximize the F1 Score.

        Parameters:
        - y_true: True binary labels.
        - y_proba: Predicted probabilities for the positive class.

        Returns:
        - optimal_threshold: The threshold that maximizes the F1 Score.
        - metrics: Dictionary containing metrics at the optimal threshold.
        """

        # Obtain predicted probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X)
        else:
            y_proba = model.predict(X)  # Fallback if no probabilities


        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # Adding epsilon to avoid division by zero
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5  # Handle edge case

        # Apply the optimal threshold
        _,_,_,metrics = evaluate_model_with_threshold(model, X, y_true, optimal_threshold, feature_names)
        metrics_df = pd.DataFrame([metrics])

        return optimal_threshold, metrics_df


    def plot_threshold_metrics(metrics_df, MODEL_NAME):
        """Plot Accuracy and Discarded Percentage against Thresholds."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metrics_df, x='Threshold', y='Accuracy', label='Accuracy')
        sns.lineplot(data=metrics_df, x='Threshold', y='Discarded Percentage', label='Discarded %')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Metric Value')
        plt.title('Accuracy and Discarded Percentage vs. Threshold')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'threshold_metrics{MODEL_NAME}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Threshold metrics plot saved to '{plot_path}'.")

    def plot_f1_score(metrics_df, MODEL_NAME):
        """Plot F1 Score against Thresholds."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metrics_df, x='Threshold', y='F1 Score', label='F1 Score')
        plt.xlabel('Probability Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Threshold')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'f1_score_threshold{MODEL_NAME}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"F1 Score plot saved to '{plot_path}'.")

    def plot_optimal_threshold(metrics_df, MODEL_NAME):
        """Identify and plot the optimal threshold based on highest F1 Score."""
        optimal_row = metrics_df.loc[metrics_df['F1 Score'].idxmax()]
        optimal_thresh = optimal_row['Threshold']
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metrics_df, x='Threshold', y='F1 Score', label='F1 Score')
        plt.axvline(x=optimal_thresh, color='red', linestyle='--', label=f'Optimal Threshold = {optimal_thresh:.2f}')
        plt.xlabel('Probability Threshold')
        plt.ylabel('F1 Score')
        plt.title('Optimal Threshold Selection')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'optimal_threshold{MODEL_NAME}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Optimal threshold plot saved to '{plot_path}'.")

    def plot_roc_curve(y, y_proba, MODEL_NAME):
        """Plot and save the ROC Curve."""
        if (((y == 1).any().any()) & ((y == 1).any().any())): 
            fpr, tpr, thresholds = roc_curve(y, y_proba)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y, y_proba):.4f})')
            plt.plot([0,1], [0,1], color='red', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.tight_layout()
            roc_path = os.path.join(PLOTS_DIR, f'roc_curve{MODEL_NAME}.png')
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC Curve saved to '{roc_path}'.")

        else:
            print(f"ROC Curve wasn't saved, because there's only one label.")

    def plot_confusion_matrix(y, y_pred, MODEL_NAME):
        """Plot and save the Confusion Matrix."""
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(PLOTS_DIR, f'confusion_matrix{MODEL_NAME}.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion Matrix saved to '{cm_path}'.")


    # ------------------------------ Configuration ------------------------------ #


    #Static paths
    DATA_DIR = os.path.join(os.getcwd(),'labeled_data')
    SAVED_MODELS_DIR = os.path.join(os.getcwd(),'saved_models')
    REPORTS_DIR = os.path.join(os.getcwd(),'reports')
    PLOTS_DIR = os.path.join(REPORTS_DIR, 'plots')
    BEST_MODELS_DIR = os.path.join(REPORTS_DIR, 'top3_best_model')



    #store all the scores so analysis can be made at the end.
    metric_results = []


    # ------------------------------ Main Testing Workflow ------------------------------ #

    #Evaluate every model that's in the directory, and creating metrics for each one.
    for delta_t in delta_t_values:

        #Dinamic Paths
        MODEL_NAME = f'final_lightgbm_model_delta_t_{delta_t}.pkl'  # Path to your saved model
        MODEL_PATH = os.path.join(SAVED_MODELS_DIR, MODEL_NAME)
        
        TEST_DATA_NAME = f"features_binary_classifierdelta_t{delta_t}1{ask_file_name}"
        TEST_DATA_PATH = os.path.join(DATA_DIR,TEST_DATA_NAME)

        #create the test file
        add_features(ask_file_name, bid_file_name, 1, y_type, delta_t)

        # Visualization Settings
        sns.set(style='whitegrid')


        # Ensure necessary directories exist
        print(f"USING MODEL {MODEL_NAME}")
        ensure_directories()
        
        # Load the trained model
        model, feature_names = load_model(MODEL_PATH)


        # # Selected Features (same as trained model features)
        # feature_names = model.named_steps['feature_names'].feature_names
        print("Features used for training:", feature_names)
        
        # Load the test data
        test_data = load_test_data(TEST_DATA_PATH)
        test_data = test_data.iloc[int((1-TEST_DATA_USED) * len(test_data)):]
        
        # Preprocess the data (using only selected features)
        X, y = preprocess_data(test_data, feature_names)
        
        if threshold_optimizer == "iterative": #This shit is currently not working (didn't try, but I know that parameters are missing),
            # Find optimal threshold           #and I don't want to deal with this shit rn, I'm not even sure if I'm going to use it.
            print("Finding the optimal threshold...")
            thresholds = np.arange(0.0, 0.5, 0.01)
            metrics_df = find_optimal_threshold(model, X, y, thresholds)

            # Identify the optimal threshold based on highest F1 Score
            optimal_metrics = metrics_df.loc[metrics_df['F1 Score'].idxmax()]
            optimal_threshold = optimal_metrics['Threshold']
            print(f"\nOptimal Threshold Selected: {optimal_threshold:.2f}")
            print(f"Metrics at Optimal Threshold:\n{optimal_metrics}")
            
            # Save metrics to a CSV for further analysis if needed (for iterative method)
            metrics_csv_path = os.path.join(REPORTS_DIR, f'threshold_metrics{MODEL_NAME}.csv')
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Threshold metrics saved to '{metrics_csv_path}'.")

            # Plot metrics (for iterative)
            plot_threshold_metrics(metrics_df, MODEL_NAME)
            plot_f1_score(metrics_df, MODEL_NAME)
            plot_optimal_threshold(metrics_df, MODEL_NAME)     
            # Evaluate and save metrics at the optimal threshold
            y_high_conf, y_pred_high_conf, y_proba_high_conf, final_metrics = evaluate_model_with_threshold(
                model, X, y, optimal_threshold, abs = True
            )  

        elif threshold_optimizer == "pr_curve":
            # Find optimal threshold using precision_recall_curve
            print("Finding the optimal threshold using precision_recall_curve...")
            optimal_threshold, metrics_df = find_optimal_threshold_pr_curve(model,X,y,feature_names)
            optimal_metrics = metrics_df

            print(f"\nOptimal Threshold Selected: {optimal_threshold:.2f}")
            print(f"Metrics at Optimal Threshold:\n{optimal_metrics}")


            # Evaluate and save metrics at the optimal threshold
            y_high_conf, y_pred_high_conf, y_proba_high_conf, final_metrics = evaluate_model_with_threshold(
                model, X, y, optimal_threshold, feature_names
            )

        metric_results.append(final_metrics)
        


        # Generate plots for high-confidence predictions
        plot_roc_curve(y_high_conf, y_proba_high_conf, MODEL_NAME)
        plot_confusion_matrix(y_high_conf, y_pred_high_conf, MODEL_NAME)
        
        # # Save final metrics to a text file --WRITES INTO THE FILE AFTER THE ENTRY BEFORE IT--
        if save_metrics == "add":
            mode = "a"
        else: # Save final metrics to a text file --CREATES A NEW METRICS FILE EVERYTHIME THE TESTING IS RAN--
            mode = "w"

        score = (
            0.5 * final_metrics['ROC AUC'] +
            0.3 * final_metrics['Accuracy'] +
            0.2 * final_metrics['Precision']  # or Recall, depending on future strategy
        )

        METRICS_FILE = os.path.join(REPORTS_DIR, f'{round(score,4)}metrics{MODEL_NAME}.txt')

        with open(METRICS_FILE, mode) as f:
            f.write("Optimal Threshold Evaluation Metrics:\n")
            f.write(f"Using: {MODEL_PATH}")
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

        


        print(f"Optimal threshold evaluation metrics appended to '{METRICS_FILE}'.")
        print("Testing completed successfully.")



    #----------------------------------- Saving n best model -----------------------------------

    metric_results_df = pd.DataFrame(metric_results)

    # Add a weighted score column based on priorities
    metric_results_df['Score'] = (
        0.5 * metric_results_df['ROC AUC'] +
        0.3 * metric_results_df['Accuracy'] +
        0.2 * metric_results_df['Precision']  # or Recall, depending on future strategy
    )

    # Sort the DataFrame by the weighted score in descending order
    metric_results_df.sort_values('Score', ascending=False, inplace=True)

    # Select the top 3 models
    top_metrics = metric_results_df.head(N_TOP_MODELS)

    # Loop through the top 3 models and copy them with the new names
    for i, (idx, row) in enumerate(top_metrics.iterrows()):
        delta_t = delta_t_values[idx]
        print(f'Rank {i+1}: Best model: {delta_t}, with the following metrics:\n{row}')

        MODEL_NAME = f'final_lightgbm_model_delta_t_{delta_t}.pkl'
        MODEL_PATH = os.path.join(SAVED_MODELS_DIR, MODEL_NAME)


        NEW_FILE_NAME = f'{round(row['Score'],4)}{MODEL_NAME}'
        NEW_MODEL_PATH = os.path.join(BEST_MODELS_DIR, NEW_FILE_NAME)
        
        if os.path.isfile(NEW_MODEL_PATH):
            print(f"THE FILE {NEW_FILE_NAME} ALREADY EXISTS")


        #copy the file to the other directory
        shutil.copy(MODEL_PATH, NEW_MODEL_PATH)
        print(f"Copied model to: {NEW_MODEL_PATH}")



