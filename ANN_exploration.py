import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout # type: ignore
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import product
from time import time
from tqdm import tqdm
from tensorflow.keras.utils import plot_model
import os
from functions import *

gpus = tf.config.experimental.list_physical_devices('GPU')
print(tf.__version__)
if gpus:
    try:
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def build_model(hidden_layers, input_dim):
    model = Sequential()
    
    # Input layer
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    
    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    
    # Output layer
    model.add(Dense(5, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load your dataset
dataset = pd.read_excel('dataset_features.xlsx')

# Extract features and target variable
X = dataset[['mean_x', 'mean_y', 'rms_x', 'rms_y', 'variance_x', 'variance_y',
            'spectral_flatness_x', 'spectral_flatness_y', 'sample_entropy_x',
       'sample_entropy_y']]
y = dataset['b4_state']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define range for number of neurons in hidden layers
min_neurons = 5
max_neurons = 9

results = []

start_time = time()

combinations = list(product(range(min_neurons, max_neurons + 1), repeat=5)) + list(product(range(min_neurons, max_neurons + 1), repeat=1)) + list(product(range(min_neurons, max_neurons + 1), repeat=2)) + list(product(range(min_neurons, max_neurons + 1), repeat=3)) + list(product(range(min_neurons, max_neurons + 1), repeat=4))

accuracy_list = []

loss_list = []

# Use tqdm to wrap the combined iterator
for index, comb in tqdm(enumerate(combinations), total=len(combinations)):
    tqdm.write(f"Training model with hidden layers: {comb}")
    # Build the model with one hidden layer
    model = build_model(comb, input_dim=X_train.shape[1])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    tqdm.write(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
    accuracy_list.append(accuracy)
    loss_list.append(loss)
    

sol_dict = {'accuracy': accuracy_list, 'loss': loss_list, 'hidden_layers': combinations}
df = pd.DataFrame(sol_dict)
df.to_excel('results.xlsx')

end_time = round((time() - start_time) / 60, 4)
        
x = 1

#os.system("shutdown /s /t 1") # Uncomment to shutdown the computer after training is complete