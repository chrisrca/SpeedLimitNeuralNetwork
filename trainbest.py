import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import keras_tuner as kt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# Load the preprocessed train and test data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # Convert to Series
y_test = pd.read_csv("y_test.csv").squeeze()    # Convert to Series

# Define the hypermodel for Keras Tuner
def build_model(hp):
    model = Sequential([
        Dense(
            units=hp.Int('units_1', min_value=512, max_value=2048, step=512),
            activation='relu',
            kernel_regularizer=l2(hp.Choice('l2_1', values=[0.01, 0.03, 0.05])),
            input_shape=(X_train.shape[1],)
        ),
        BatchNormalization(),
        Dropout(hp.Choice('dropout_1', values=[0.2, 0.3, 0.4])),
        Dense(
            units=hp.Int('units_2', min_value=256, max_value=1024, step=256),
            activation='relu',
            kernel_regularizer=l2(hp.Choice('l2_2', values=[0.01, 0.03, 0.05]))
        ),
        BatchNormalization(),
        Dropout(hp.Choice('dropout_2', values=[0.2, 0.3, 0.4])),
        Dense(1)  # Output layer
    ])
    
    # Learning rate scheduler for training
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=hp.Choice('learning_rate', values=[0.001, 0.0005, 0.0001]),
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-6
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mse',
        metrics=['mae']
    )
    return model

# Reload the tuner from the existing directory
tuner = kt.Hyperband(
    build_model,  # Make sure build_model is defined exactly as it was during the initial tuning
    objective='val_loss',
    max_epochs=50,
    factor=4,
    directory='my_dir',
    project_name='speed_limit_nn_hyperband'
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters Found:")
for key, value in best_hps.values.items():
    print(f"{key}: {value}")

# Build the model using the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Path to the checkpoint directory
checkpoint_filepath = "model_checkpoints/checkpoint"

# Load the checkpoint weights if they exist
if os.path.exists(checkpoint_filepath):
    print("Loading weights from checkpoint...")
    best_model.load_weights(checkpoint_filepath)

# Compile the model (adjust learning rate if needed)
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Use a smaller learning rate for fine-tuning
    loss='mse',
    metrics=['mae']
)

# Train the model further
history = best_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1000,  # Additional epochs for fine-tuning
    batch_size=512,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="model_checkpoints/best_model_checkpoint",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs/final_training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ]
)

# Evaluate the model on training and testing data
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print("\nFinal Training Metrics:")
print("Training Data:")
print(f"  RMSE: {mean_squared_error(y_train, y_pred_train, squared=False):.2f}")
print(f"  MAE: {mean_absolute_error(y_train, y_pred_train):.2f}")
print(f"  R2: {r2_score(y_train, y_pred_train):.2f}")

print("\nTesting Data:")
print(f"  RMSE: {mean_squared_error(y_test, y_pred_test, squared=False):.2f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"  R2: {r2_score(y_test, y_pred_test):.2f}")

# Plot training vs validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss (Fine-Tuning)')
plt.legend()
plt.show()

# Save the fine-tuned model
final_model_path = "best_fine_tuned_model.h5"
best_model.save(final_model_path)
print(f"Fine-tuned model saved as '{final_model_path}'.")
