import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the data
data = pd.read_csv("test_merged.csv")

# Define features and target
features = ['highway', 'total_length', 'curvature', 'avg_surrounding_speed_limit', 'total_surrounding_length']
target = 'speed_limit'

# Encode the 'highway' categorical feature
encoder = LabelEncoder()
data['highway'] = encoder.fit_transform(data['highway'])

# Scale the numerical features
scaler = StandardScaler()
data[features[1:]] = scaler.fit_transform(data[features[1:]])

# Prepare the feature set and target
X = data[features]
y = data[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Round predictions to the nearest multiple of 5
rounded_predictions_train = np.round(y_pred_train / 5) * 5
rounded_predictions_test = np.round(y_pred_test / 5) * 5

# Evaluate the model
train_rmse = mean_squared_error(y_train, rounded_predictions_train, squared=False)
train_mae = mean_absolute_error(y_train, rounded_predictions_train)
train_r2 = r2_score(y_train, rounded_predictions_train)

test_rmse = mean_squared_error(y_test, rounded_predictions_test, squared=False)
test_mae = mean_absolute_error(y_test, rounded_predictions_test)
test_r2 = r2_score(y_test, rounded_predictions_test)

print("Training Metrics:")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  MAE: {train_mae:.2f}")
print(f"  R2: {train_r2:.2f}")

print("\nTesting Metrics:")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE: {test_mae:.2f}")
print(f"  R2: {test_r2:.2f}")

# Save predictions
X_test['actual_speed_limit'] = y_test
X_test['predicted_speed_limit'] = rounded_predictions_test
X_test.to_csv("linear_regression_predictions.csv", index=False)
print("Predictions saved to 'linear_regression_predictions.csv'")
