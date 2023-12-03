import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv("data.csv")

# Separate features and target variable
X = data[['Year_Recor', 'AADT', 'IRI', 'ROUTE_QUAL', 'TRUCK']]
y = data['TRAFFIC_JAM']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create sequences for LSTM
sequence_length = 10  # You can adjust this sequence length
X_train_sequence = []
X_test_sequence = []

for i in range(sequence_length, len(X_train)):
    X_train_sequence.append(X_train[i - sequence_length:i])
for i in range(sequence_length, len(X_test)):
    X_test_sequence.append(X_test[i - sequence_length:i])

X_train_sequence = np.array(X_train_sequence)
X_test_sequence = np.array(X_test_sequence)


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_sequence.shape[1], X_train_sequence.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train_sequence, y_train[sequence_length:], epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_sequence, y_test[sequence_length:])
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
