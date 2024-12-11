import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
import joblib

os.chdir("C:/Users/ppoli/Desktop/einstein/4_semestre/PBL/neural_net")

# Ensure the neural_net directory exists
neural_net_dir = 'neural_net'
if not os.path.exists(neural_net_dir):
    os.makedirs(neural_net_dir)

# Define the number of features
num_features = 10  # Since we have 10 bands for mao and ant separately

# ----------------- Create Dummy Hand Model (Mão) -----------------

# Create a dummy scaler for mao
scaler_mao = StandardScaler()
# Fit the scaler on dummy data
scaler_mao.fit(np.random.rand(100, num_features))
# Save the scaler
joblib.dump(scaler_mao, os.path.join(neural_net_dir, 'scaler_mao.save'))

# Create a dummy model for mao
model_mao = Sequential()
model_mao.add(Input(shape=(num_features,)))
model_mao.add(Dense(64, activation='relu'))
model_mao.add(Dense(32, activation='relu'))
model_mao.add(Dense(4, activation='softmax'))
# Since it's a dummy model, we can skip training and just save it
model_mao.save(os.path.join(neural_net_dir, 'trained_model_mao.keras'))

print("Dummy hand model and scaler have been saved to neural_net directory.")

# ----------------- Create Dummy Forearm Model (Antebraço) -----------------

# Create a dummy scaler for ant
scaler_ant = StandardScaler()
# Fit the scaler on dummy data
scaler_ant.fit(np.random.rand(100, num_features))
# Save the scaler
joblib.dump(scaler_ant, os.path.join(neural_net_dir, 'scaler_ant.save'))

# Create a dummy model for ant
model_ant = Sequential()
model_ant.add(Input(shape=(num_features,)))
model_ant.add(Dense(64, activation='relu'))
model_ant.add(Dense(32, activation='relu'))
model_ant.add(Dense(4, activation='softmax'))
# Since it's a dummy model, we can skip training and just save it
model_ant.save(os.path.join(neural_net_dir, 'trained_model_ant.keras'))

print("Dummy forearm model and scaler have been saved to neural_net directory.")
