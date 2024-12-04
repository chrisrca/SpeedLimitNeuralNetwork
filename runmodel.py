import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('best_fine_tuned_model.h5')

# Load the new data
new_data = pd.read_csv("test_merged.csv")  # Replace with your file path

# Extract relevant columns for preprocessing
features = ['highway', 'total_length', 'curvature', 'avg_surrounding_speed_limit', 'total_surrounding_length']
way_id_column = 'way_id'

# Encode the 'highway' categorical feature
encoder = LabelEncoder()
new_data['highway'] = encoder.fit_transform(new_data['highway'])

# Scale the numerical features
scaler = StandardScaler()
new_data[features[1:]] = scaler.fit_transform(new_data[features[1:]])

# Prepare the feature set for prediction
X_new = new_data[features]

# Run 20 iterations and average predictions
predictions = []
for _ in range(3):
    pred = model.predict(X_new)
    predictions.append(pred)

# Convert predictions to a NumPy array and compute the average
predictions = np.mean(predictions, axis=0)

# Round predictions to the nearest multiple of 5
rounded_predictions = np.round(predictions / 5) * 5

# Add predictions to the DataFrame
new_data['predicted_speed_limit'] = rounded_predictions

# Create a new DataFrame with `way_id` and `predicted_speed_limit`
output_data = new_data[[way_id_column, 'predicted_speed_limit']]

# Save the results to a new CSV
output_data.to_csv("predicted_speed_limits.csv", index=False)

print("Predictions saved to 'predicted_speed_limits.csv'")
