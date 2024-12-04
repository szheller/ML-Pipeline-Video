import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ray

# Load provided_data.csv
data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# Convert 'effort' column to numeric; non-numeric entries will be set to NaN
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# Impute missing 'effort' values using linear interpolation
data['effort'] = data['effort'].interpolate(method='linear')

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Load target.csv
target = pd.read_csv('target.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged1 = pd.merge(data, target, on='frame', how='inner')

# Load sudden_change.csv
sudden_change = pd.read_csv('sudden_change.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
sudden_change['frame'] = sudden_change['frame'].astype(int)

# Rename value column to sudden_change to be unique from target
sudden_change.columns = ['frame', 'sudden_change']

# Merge data and target on 'frame'
merged = pd.merge(merged1, sudden_change, on='frame', how='inner')

# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort', 'sudden_change']
print(features)
X = merged[features]
y = merged['value']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create lag features for time series data
def create_lag_features(X, window_size):
    X_lagged = pd.DataFrame()
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
    return X_lagged.dropna()

# Train final model with best parameters
best_window_size = 100
best_n_estimators = 10

X_lagged = create_lag_features(X_scaled, best_window_size)
y_lagged = y.iloc[best_window_size - 1:]
frames_lagged = merged['frame'].iloc[best_window_size - 1:]

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# Split into train and test sets
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]

# Train final model
clf = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
# clf = AdaBoostClassifier(n_estimators=best_n_estimators, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Compute and print classification report
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)

# Write predictions to CSV with the same syntax as target.csv
predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred})
predictions_df.to_csv('predictions.csv', index=False)

