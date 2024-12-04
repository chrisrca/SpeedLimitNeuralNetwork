# Neural Network trained on road data to estimate speed limits

This project builds a deep learning model for predicting speed limits, using TensorFlow, Keras Tuner for hyperparameter optimization, and custom callbacks for training management.

# Key Metrics Interpretation

## Training Data Metrics:
      RMSE (Root Mean Squared Error): 3.68
      MAE (Mean Absolute Error): 1.67
      R² (Coefficient of Determination): 0.86

## Testing (Validation) Data Metrics:
      RMSE: 3.69
      MAE: 1.67
      R²: 0.86

## What It Means:

  The model has similar performance on both the training and testing datasets indicating that the model generalizes well to unseen data.

  The R² score of 0.86 suggests that the model explains 86% of the variance in the target variable, which is adequate for this regression problem.

  The small difference between RMSE and MAE indicates the model has not been overly influenced by large errors (outliers), as RMSE is typically more sensitive to outliers than MAE.

## Additional Insights:

  When tested on real-world data, the model yielded results ≤ 10 mph of the actual speed limits, demonstrating its practical utility for estimating speed limits based on road data. However, incorporating more extensive data, such as elevation and other road-related information, could potentially improve the model's accuracy and predictive power.
