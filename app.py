import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# ----------------------------------
# Create Sequences for GRU
# ----------------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting Close price only
    return np.array(X), np.array(y)

# ----------------------------------
# Train GRU Model
# ----------------------------------
def train_gru_model(data, seq_length=60):
    X, y = create_sequences(data, seq_length)
    model = Sequential()
    model.add(GRU(units=64, return_sequences=False, input_shape=(seq_length, data.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)
    return model, X, y

# ----------------------------------
# Forecast Future Values
# ----------------------------------
def forecast(model, data, forecast_days, scaler, seq_length=60):
    prediction = []
    input_seq = data[-seq_length:].copy()

    for _ in range(forecast_days):
        pred = model.predict(input_seq.reshape(1, seq_length, data.shape[1]), verbose=0)[0][0]
        next_row = input_seq[-1].copy()
        next_row[0] = pred  # update close price
        input_seq = np.vstack((input_seq[1:], next_row))
        prediction.append(pred)

    dummy_rows = np.tile(data[-1], (forecast_days, 1))
    dummy_rows[:, 0] = prediction  # replace Close price only
    prediction_rescaled = scaler.inverse_transform(dummy_rows)[:, 0]
    return prediction_rescaled

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("📈 Stock Price Forecast")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())

if st.button("Predict"):
    data = load_data(ticker, start_date, end_date)
    if data is None or data.empty:
        st.warning("No data found for the selected ticker.")
        st.stop()

    df_features = data[['Close', 'Open', 'High', 'Low', 'Volume']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)

    seq_length = 60
    model, X, y = train_gru_model(scaled_data, seq_length=seq_length)
    prediction = forecast(model, scaled_data, forecast_days=30, scaler=scaler, seq_length=seq_length)

    forecast_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Close': prediction})

    # Chart 1: Actual vs Model Predicted (Interactive)
    st.subheader("📊 Actual vs Model Predicted (Interactive)")
    model_pred = model.predict(X, verbose=0)

    dummy_model_rows = np.tile(scaled_data[-1], (model_pred.shape[0], 1))
    dummy_model_rows[:, 0] = model_pred.flatten()
    model_pred_prices = scaler.inverse_transform(dummy_model_rows)[:, 0]
    model_pred_dates = data['Date'][seq_length:seq_length + len(model_pred_prices)]

    # Model Evaluation Metrics
    actual_prices = data['Close'][seq_length:seq_length + len(model_pred_prices)].values.flatten()
    predicted_prices = model_pred_prices.flatten()

    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    accuracy = 100 - mape
    st.write(f"**Accuracy:** {accuracy:.2f}%")
    
        # mae = mean_absolute_error(actual_prices, predicted_prices)
        # mse = mean_squared_error(actual_prices, predicted_prices)
        # rmse = np.sqrt(mse)
        # r2 = r2_score(actual_prices, predicted_prices)

        # st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
        # st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
        # st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
        # st.write(f"**R² Score:** {r2:.4f}")

    df_plot = pd.DataFrame({
        'Date': model_pred_dates.reset_index(drop=True),
        'Actual': actual_prices
    })
    df_model = pd.DataFrame({
        'Date': model_pred_dates.reset_index(drop=True),
        'Predicted': predicted_prices
    })
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Actual'], mode='lines', name='Actual', line=dict(color='royalblue')))
    fig1.add_trace(go.Scatter(x=df_model['Date'], y=df_model['Predicted'], mode='lines', name='Predicted', line=dict(dash='dot', color='firebrick')))
    fig1.update_layout(title=f"{ticker} - Actual vs Model Predicted", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig1)

    # Chart 2: Forecasted Price (Interactive)
    st.subheader("🔮 Forecasted Price (Next 30 Days)")
    fig2 = px.line(forecast_df, x='Date', y='Predicted Close', title=f'{ticker} - 30 Day Forecast')
    st.plotly_chart(fig2)

    st.subheader("📋 Forecasted Data")
    st.dataframe(forecast_df)