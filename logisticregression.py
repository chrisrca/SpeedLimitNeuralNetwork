import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# Load the data
data = pd.read_csv("test_merged.csv")

# Define features and target
features = ['highway', 'total_length', 'curvature', 'avg_surrounding_speed_limit', 'total_surrounding_length']
target = 'speed_limit'

# Create speed limit categories in 5 MPH intervals
bins = np.arange(0, data[target].max() + 5, 5)  # Ensure bins cover all speed limits
labels = bins[:-1]  # Labels correspond to bin lower bounds
data['speed_limit_category'] = pd.cut(data[target], bins=bins, labels=labels, right=False)

# Drop rows with NaN in the target column
data = data.dropna(subset=['speed_limit_category'])

# Ensure the target column is numeric
data['speed_limit_category'] = data['speed_limit_category'].astype(int)

# Encode the 'highway' categorical feature
encoder = LabelEncoder()
data['highway'] = encoder.fit_transform(data['highway'])

# Scale the numerical features
scaler = StandardScaler()
data[features[1:]] = scaler.fit_transform(data[features[1:]])

# Prepare the feature set and target
X = data[features]
y = data['speed_limit_category']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Metrics:")
print(f"  Accuracy: {train_accuracy:.2f}")

print("\nTesting Metrics:")
print(f"  Accuracy: {test_accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(conf_matrix)

# Save predictions
X_test['actual_speed_limit_category'] = y_test
X_test['predicted_speed_limit_category'] = y_pred_test
X_test.to_csv("logistic_regression_predictions_5mph_intervals.csv", index=False)
print("Predictions saved to 'logistic_regression_predictions_5mph_intervals.csv'")
