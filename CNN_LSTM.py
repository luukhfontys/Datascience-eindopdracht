import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout # type: ignore
import pandas as pd
from functions import *

def collect_X_train_CNN_LSTM():
    data_path = 'train'
    X_train_list = []
    for i in range(0, 1482):
        data = load_data_index(data_path, 4, i)
        X_train_list.append(data)
        
    X_train = np.array(X_train_list)
    
    return X_train


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
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(20480, 2)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# LSTM layers
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))

# Fully connected layers
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))  # Adjusted to 5 wear stages

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

X_train = collect_X_train_CNN_LSTM()
y_train = pd.read_csv('train/bearing_conditions.csv', delimiter=';')['b4_state'].to_numpy()[:1482]


history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
model.save('C:/VSCode/Datascience-eindopdracht/model.keras')