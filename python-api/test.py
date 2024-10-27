import pandas as pd
import numpy as np
import pickle
import random

# Load the test dataset
test_data_path = '../model/test.csv'
df_test = pd.read_csv(test_data_path)

# Load the saved model
with open('device_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Select 10 random rows from the test dataset
random_indices = random.sample(range(len(df_test)), 10)
random_devices = df_test.iloc[random_indices]

# Display the selected devices
print("Randomly selected devices from test dataset:")
print(random_devices)

# Prepare the data for prediction (scaling should be done similarly as training)
from sklearn.preprocessing import StandardScaler

# Assuming the scaler used in training is also saved, you can load it
scaler = StandardScaler()
df_train = pd.read_csv('../model/train.csv')  # Load train data to fit the scaler

# Remove the 'id' column if present in training data
if 'id' in df_train.columns:
    df_train = df_train.drop(columns=['id'])

X_train = df_train.drop(columns=['price_range'])
scaler.fit(X_train)  # Fit the scaler on training data features

# Scale the selected random devices (exclude 'id' if present)
if 'id' in random_devices.columns:
    random_devices = random_devices.drop(columns=['id'])

random_devices_scaled = scaler.transform(random_devices)

# Make predictions
predictions = model.predict(random_devices_scaled)

# Display the predictions
print("\nPredictions for the selected devices:")
for idx, prediction in zip(random_indices, predictions):
    print(f"Device ID {df_test.iloc[idx]['id']}: Predicted Price Range -> {prediction}")
