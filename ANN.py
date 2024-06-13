import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout # type: ignore
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
        
model = Sequential()

# Convolutional layers
model.add(Dense(10, input_dim=19, activation='relu'))

model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
print(model.summary())

dataset = pd.read_excel('dataset_features.xlsx')

X = dataset[['index', 'mean_x', 'mean_y', 'std_x', 'std_y', 'rms_x', 'rms_y', 'kurtosis_x', 'kurtosis_y', 'variance_x', 'variance_y',
       'crest_factor_x', 'crest_factor_y', 'skewness_x', 'skewness_y', 'spectral_flatness_x', 'spectral_flatness_y', 'sample_entropy_x',
       'sample_entropy_y']]

y = dataset['b4_state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

x=1