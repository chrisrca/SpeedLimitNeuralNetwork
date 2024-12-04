import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
test_data = pd.read_csv("test_merged.csv")
predicted_data = pd.read_csv("predicted_speed_limits.csv")

# Merge the test data with predicted data on `way_id`
merged_data = pd.merge(test_data, predicted_data, on="way_id")

# Filter out rows where the actual speed limit is 0 mph
filtered_data = merged_data[merged_data['speed_limit'] != 0]

# Calculate evaluation metrics
rmse = mean_squared_error(filtered_data['speed_limit'], filtered_data['predicted_speed_limit'], squared=False)
mae = mean_absolute_error(filtered_data['speed_limit'], filtered_data['predicted_speed_limit'])
r2 = r2_score(filtered_data['speed_limit'], filtered_data['predicted_speed_limit'])

# Print the results
print("Evaluation Metrics (Ignoring 0 mph speed limits):")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  R2 Score: {r2:.2f}")

# Optional: Save the filtered results for inspection
filtered_data.to_csv("filtered_results.csv", index=False)
print("Filtered results saved to 'filtered_results.csv'")
