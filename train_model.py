import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("creditcard.csv").sample(n=50000, random_state=42)

# Features and Target
X = data.drop("Class", axis=1)
y = data["Class"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50),
    "XGBoost": XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(iterations=50, verbose=0)
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Accuracy: {score}")
    
    if score > best_score:
        best_score = score
        best_model = model

# Save best model
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully!")
