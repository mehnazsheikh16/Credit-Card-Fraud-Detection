import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import joblib
import os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), '../../data/creditcard.csv')
df = pd.read_csv(data_path)

# Handle missing values (if any)
df = df.dropna()

# Handle class imbalance (undersample majority class)
df_majority = df[df.Class == 0]
df_minority = df[df.Class == 1]
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority)*5, # 5:1 ratio
                                   random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

# Feature/target split
X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=3, scoring='recall', verbose=2)
grid.fit(X_train, y_train)

# Evaluation
y_pred = grid.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Save model
model_path = os.path.join(os.path.dirname(__file__), 'rf_model.joblib')
joblib.dump(grid.best_estimator_, model_path)
print(f"Model saved to {model_path}")
