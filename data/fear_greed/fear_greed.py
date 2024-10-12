import pandas as pd
import json

# Load the JSON file
file_path = 'fear_greed.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract the current fear and greed data
current_data = data['fear_and_greed']
df_current = pd.DataFrame([current_data])

# Extract the historical fear and greed data
historical_data = data['fear_and_greed_historical']['data']
df_historical = pd.DataFrame(historical_data)

# Convert timestamp columns to datetime
df_current['timestamp'] = pd.to_datetime(df_current['timestamp'])
df_current['previous_close'] = pd.to_numeric(df_current['previous_close'], errors='coerce')
df_current['previous_1_week'] = pd.to_numeric(df_current['previous_1_week'], errors='coerce')
df_current['previous_1_month'] = pd.to_numeric(df_current['previous_1_month'], errors='coerce')
df_current['previous_1_year'] = pd.to_numeric(df_current['previous_1_year'], errors='coerce')

df_historical['date'] = pd.to_datetime(df_historical['x'], unit='ms')
df_historical = df_historical.rename(columns={'y': 'score', 'rating': 'rating'})
df_historical = df_historical.drop(columns='x')

# Reorder columns for clarity
df_historical = df_historical[['date', 'score', 'rating']]

# Save current data to CSV
df_current.to_csv('fear_greed_current.csv', index=False)

# Save historical data to CSV
df_historical.to_csv('fear_greed_historical.csv', index=False)

print("Data has been saved to CSV files.")
