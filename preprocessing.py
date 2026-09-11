import pandas as pd
import numpy as np

# Load ad click data
df = pd.read_csv('data/ad_click_dataset.csv').copy()

print("Dataset loaded!")
print(f"Shape: {df.shape}")

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())
df['gender'] = df['gender'].fillna('Unknown')
df['device_type'] = df['device_type'].fillna('Unknown')
df['browsing_history'] = df['browsing_history'].fillna('Unknown')
df['time_of_day'] = df['time_of_day'].fillna('Unknown')
df['ad_position'] = df['ad_position'].fillna('Unknown')

# Save for next step
df.to_csv('data/ad_data_processed.csv', index=False)
print("Data processed and saved!")
