import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Load data
df = pd.read_csv('data/ad_data_processed.csv').copy()

print("=== Ad Click Prediction ===\n")

# Encode categorical variables
categorical_cols = ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
features = ['age', 'gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']
X = df[features]
y = df['click']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
print("Logistic Regression:")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
print(f"  AUC-ROC: {lr_auc:.3f}")
print(f"  Accuracy: {lr_acc:.3f}\n")

# Random Forest
print("Random Forest:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
print(f"  AUC-ROC: {rf_auc:.3f}")
print(f"  Accuracy: {rf_acc:.3f}\n")

# Best model
best_model = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"
print(f"Best Model: {best_model}")
