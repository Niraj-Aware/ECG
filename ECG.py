!pip install pandas streamlit plotly scikit-learn tensorflow pyserial

import numpy as np
from scipy.signal import resample

def preprocess_ecg_data(data):
    # Resample the data to have a fixed length of 187
    data_resampled = resample(data, 187)
    
    # Normalize the data to have zero mean and unit variance
    data_normalized = (data_resampled - np.mean(data_resampled)) / np.std(data_resampled)
    
    # Expand the dimensions of the data to have a shape of (1, 187, 1)
    data_expanded = np.expand_dims(data_normalized, axis=(0, -1))
    
    return data_expanded

import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

def predict_disease(data):
    preprocessed_data = preprocess_ecg_data(data)
    predictions = model.predict(preprocessed_data)
    disease = np.argmax(predictions)
    
    return disease

import streamlit as st
import serial

st.title('Real-Time ECG Disease Detection')

# Define the serial port and baud rate
port = 'COM3'
baud_rate = 115200

# Connect to the serial port
ser = serial.Serial(port, baud_rate)

# Create a variable to store the ECG data
ecg_data = []

# Create a variable to store the disease prediction
disease_prediction = None

# Create a variable to store the previous disease prediction
previous_disease_prediction = None

while True:
    # Read the ECG data from the serial port
    data = ser.readline().decode('utf-8').rstrip()
    
    # Split the data into individual values
    values = data.split(',')
    
    # Convert the values to float and append them to the ecg_data list
    ecg_data.append(float(values[0]))
    
    # If the ecg_data list has a length of 187, predict the disease and update the disease_prediction variable
    if len(ecg_data) == 187:
        disease_prediction = predict_disease(ecg_data)
        ecg_data = []
    
    # If the disease prediction has changed, update the previous_disease_prediction variable and print the new prediction
    if disease_prediction != previous_disease_prediction:
        previous_disease_prediction = disease_prediction
        st.write('Disease Prediction:', disease_prediction)
