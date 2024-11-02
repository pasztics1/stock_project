import joblib 
import os

def store_feature_names(X, y=None):
    if hasattr(X, 'columns'):
        store_feature_names.feature_names = X.columns.tolist()
    return X

model_filepath = '0.6095final_lightgbm_model_delta_t_32.pkl'


#OLD VERSION
# # filepath = os.path.join(os.getcwd(),'reports')
# # filepath2 = os.path.join(filepath, 'top3_best_model')
# # model_filepath = os.path.join(filepath2,model_name)


# loaded_pipeline = joblib.load(model_filepath)


# feature_names = loaded_pipeline.named_steps['feature_names'].feature_names
# print(feature_names)

#NEW VERSION

"""Load the trained model from a .pkl file using joblib."""
# Retrieve feature names from the pipeline
model = joblib.load(model_filepath)
feature_names = model.named_steps['model'].original_feature_names
print(f"Model loaded successfully from '{model_filepath}'.")
print(f"Feature Names: {feature_names}")


#OLD WORKING COPY WHERE THE COLUMN NAMES COULD ACTUALLY BE RETRIEVED

# def objective(trial, X, y, tscv):
#     """
#     Objective function for Optuna to optimize LightGBM hyperparameters within a Pipeline.
#     """
#     param = {
#         'model__objective': 'binary',
#         'model__metric': 'binary_logloss',
#         'model__boosting_type': 'gbdt',
#         'model__verbosity': -1,
#         'model__random_state': 42,
#         'model__n_estimators': trial.suggest_int('n_estimators', 500, 5000),
#         'model__learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
#         'model__max_depth': trial.suggest_int('max_depth', 5, 100),
#         'model__num_leaves': trial.suggest_int('num_leaves', 20, 300),
#         'model__min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 500),
#         'model__feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#         'model__bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#         'model__bagging_freq': trial.suggest_int('bagging_freq', 1, 20),
#         'model__lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#         'model__lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#         'model__bagging_seed': trial.suggest_int('bagging_seed', 0, 100),
#         'model__class_weight': 'balanced'
#     }

#     # Define the pipeline with scaling, SMOTE, and the model
#     pipeline = ImbPipeline([
#         ('scaler', StandardScaler()),
#         ('smote', SMOTE(random_state=42)),
#         ('model', lgb.LGBMClassifier())
#     ])

#     # Set the parameters in the pipeline
#     pipeline.set_params(**param)

#     roc_auc_scores = []

#     for train_index, test_index in tscv.split(X):
#         X_train, X_valid = X[train_index], X[test_index]
#         y_train, y_valid = y[train_index], y[test_index]

#         pipeline.fit(
#             X_train, y_train,
#             model__eval_set=[(X_valid, y_valid)],
#             model__eval_metric='binary_logloss',
#             model__callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
#         )

#         y_pred_prob = pipeline.predict_proba(X_valid)[:, 1]
#         roc_auc = roc_auc_score(y_valid, y_pred_prob)
#         roc_auc_scores.append(roc_auc)

#     return np.mean(roc_auc_scores)



# for train_index, test_index in tscv_outer.split(X):
#     outer_fold += 1
#     print(f"\n--- Outer Fold {outer_fold} ---")
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     # Inner CV for hyperparameter tuning is already done, use best_params
#     pipeline = ImbPipeline([
#         ('feature_names', FunctionTransformer(store_feature_names, validate=False)),
#         ('scaler', StandardScaler()),
#         ('smote', SMOTE(random_state=42)),
#         ('model', lgb.LGBMClassifier(**best_params, random_state=42, class_weight='balanced'))
#     ])
#     # Train the model
#     pipeline.fit(X_train, y_train)
#     # After fitting the pipeline, manually add the feature names
#     pipeline.named_steps['feature_names'].feature_names = included_features



# # ===========================
# # Step 6: Save the Model
# # ===========================
# print("\nStep 6: Saving the Trained Model")
# model_filename = f'final_lightgbm_model_delta_t_{delta_t}.pkl'
# model_filepath = os.path.join(model_output_path, model_filename)
# joblib.dump(pipeline, model_filepath)
# print(f"Model saved at {model_filepath}")