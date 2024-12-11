import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Define the neural_net directory
neural_net_dir = 'neural_net'

# Ensure the neural_net directory exists
if not os.path.exists(neural_net_dir):
    os.makedirs(neural_net_dir)

# Load the dataset from neural_net directory
data = pd.read_csv(os.path.join(neural_net_dir, 'training_data.csv'), sep=';')


# ----------------- Hand Model (Mão) -----------------

# Prepare the features and labels for the hand model
mao_features = [f"mao_b{i}" for i in range(1, 11)]
X_mao = data[mao_features].values
y_mao = data['classe_mao'].values

# One-hot encode the labels
y_mao_encoded = to_categorical(y_mao, num_classes=4)

# Split into training and test sets
X_mao_train, X_mao_test, y_mao_train, y_mao_test = train_test_split(
    X_mao, y_mao_encoded, test_size=0.2, random_state=42
)

# Standardize the features
scaler_mao = StandardScaler()
X_mao_train_scaled = scaler_mao.fit_transform(X_mao_train)
X_mao_test_scaled = scaler_mao.transform(X_mao_test)

# Define the neural network model for hand
model_mao = Sequential()
model_mao.add(Dense(64, input_dim=X_mao_train_scaled.shape[1], activation='relu'))
model_mao.add(Dense(32, activation='relu'))
model_mao.add(Dense(4, activation='softmax'))

# Compile the model
model_mao.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_mao.fit(X_mao_train_scaled, y_mao_train, epochs=50, batch_size=32, validation_data=(X_mao_test_scaled, y_mao_test))

# Save the trained hand model and scaler in neural_net directory
model_mao.save(os.path.join(neural_net_dir, 'trained_model_mao.h5'))
joblib.dump(scaler_mao, os.path.join(neural_net_dir, 'scaler_mao.save'))

print("Hand model and scaler have been saved to neural_net directory.")

# ----------------- Forearm Model (Antebraço) -----------------

# Prepare the features and labels for the forearm model
ant_features = [f"ant_b{i}" for i in range(1, 11)]
X_ant = data[ant_features].values
y_ant = data['classe_ant'].values

# One-hot encode the labels
y_ant_encoded = to_categorical(y_ant, num_classes=4)

# Split into training and test sets
X_ant_train, X_ant_test, y_ant_train, y_ant_test = train_test_split(
    X_ant, y_ant_encoded, test_size=0.2, random_state=42
)

# Standardize the features
scaler_ant = StandardScaler()
X_ant_train_scaled = scaler_ant.fit_transform(X_ant_train)
X_ant_test_scaled = scaler_ant.transform(X_ant_test)

# Define the neural network model for forearm
model_ant = Sequential()
model_ant.add(Dense(64, input_dim=X_ant_train_scaled.shape[1], activation='relu'))
model_ant.add(Dense(32, activation='relu'))
model_ant.add(Dense(4, activation='softmax'))

# Compile the model
model_ant.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_ant.fit(X_ant_train_scaled, y_ant_train, epochs=50, batch_size=32, validation_data=(X_ant_test_scaled, y_ant_test))

# Save the trained forearm model and scaler in neural_net directory
model_ant.save(os.path.join(neural_net_dir, 'trained_model_ant.h5'))
joblib.dump(scaler_ant, os.path.join(neural_net_dir, 'scaler_ant.save'))

print("Forearm model and scaler have been saved to neural_net directory.")
