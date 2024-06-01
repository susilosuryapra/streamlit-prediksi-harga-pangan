import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm_model(data, prediction_months=1):
    # Preprocessing data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['harga'].values.reshape(-1, 1))
    
    # Create training data
    prediction_days = prediction_months * 30
    if len(scaled_data) < prediction_days:
        raise ValueError("Data tidak cukup panjang untuk prediksi {} bulan. Harap gunakan data yang lebih panjang.".format(prediction_months))
    
    X_train, y_train = [], []
    
    for i in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    return model, scaler

st.title('Prediksi Harga Bahan Pokok')
st.write('Upload dataset dan pilih durasi prediksi untuk melihat hasil.')

# Upload CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    
    # Convert 'Date' column to datetime with correct format
    try:
        data['periode'] = pd.to_datetime(data['periode'], format='%d/%m/%Y')
    except ValueError:
        data['periode'] = pd.to_datetime(data['periode'], format='%b-%y')
    
    st.write(data.head())
    
    # Slider untuk durasi prediksi
    prediction_months = st.slider('Pilih durasi prediksi (bulan)', 1, 4, 1)
    
    if st.button('Prediksi'):
        try:
            # Train model
            model, scaler = train_lstm_model(data, prediction_months)
            
            # Create testing data
            prediction_days = prediction_months * 30
            last_days = data['harga'][-prediction_days:].values
            last_days_scaled = scaler.transform(last_days.reshape(-1, 1))
            
            X_test = []
            X_test.append(last_days_scaled)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Predict
            predicted_price = model.predict(X_test)
            predicted_price = scaler.inverse_transform(predicted_price)
            
            st.write(f"Prediksi harga dalam {prediction_months} bulan mendatang: {predicted_price[0][0]}")
            
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(data['periode'].values, data['harga'].values, color='blue', label='Harga Aktual')
            future_dates = pd.date_range(start=data['periode'].iloc[-1], periods=prediction_months*30 + 1, freq='D')
            future_prices = np.concatenate(([data['harga'].values[-1]], predicted_price.flatten()))
            
            # Ensure that future_prices and future_dates have the same length
            future_dates = future_dates[:len(future_prices)]
            
            plt.plot(future_dates, future_prices, color='red', linestyle='--', label='Prediksi Harga')
            plt.xlabel('Tanggal')
            plt.ylabel('Harga')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
        except ValueError as e:
            st.error(e)
