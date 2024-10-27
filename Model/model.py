import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Step 1: Load the Dataset
train_data_path = 'train.csv'
test_data_path = 'test.csv'

df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)

# Handling the 'id' column which exists only in the training data
if 'id' in df_train.columns:
    df_train = df_train.drop(columns=['id'])

# Features and target
X = df_train.drop(columns=['price_range'])
y = df_train['price_range']

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 2: Model Selection and Hyperparameter Tuning using GridSearchCV
# Choosing XGBoost Classifier for its efficiency and performance
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Defining the parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid Search to find the best parameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best estimator
best_xgb_model = grid_search.best_estimator_

# Model Evaluation
preds = best_xgb_model.predict(X_val)
print("Confusion Matrix:")
print(confusion_matrix(y_val, preds))
print("Classification Report:")
print(classification_report(y_val, preds))
print("Accuracy:", accuracy_score(y_val, preds))

# Saving the tuned model using Pickle
with open('device_price_model.pkl', 'wb') as model_file:
    pickle.dump(best_xgb_model, model_file)

print("Tuned XGBoost model has been saved successfully.")
