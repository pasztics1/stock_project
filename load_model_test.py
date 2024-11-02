import joblib

# Define the store_feature_names function
def store_feature_names(X, y=None):
    return X

# Now, load the model
delta_t = 36
MODEL_PATH = f'final_lightgbm_model_delta_t_{delta_t}.pkl'

loaded_pipeline = joblib.load(MODEL_PATH)
feature_names = loaded_pipeline.named_steps['feature_names'].feature_names
print("Features used for training:", feature_names)
