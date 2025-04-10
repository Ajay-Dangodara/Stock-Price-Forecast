# 📈 Stock Price Forecasting with GRU

This Streamlit web app predicts stock prices using a GRU (Gated Recurrent Unit) neural network model. It uses historical data from Yahoo Finance and forecasts future stock prices for the next 30 days.

## 🚀 Features

- Load historical stock data using Yahoo Finance.
- Preprocess data using MinMaxScaler.
- Build and train a GRU model.
- Predict and visualize future stock prices.
- Display model Accuracy (%)
- Interactive Plotly charts.
- Clean and responsive Streamlit UI.

## 📦 Installation

Follow these steps to install and run the application:

1. **Clone the repository:**
   ```bash
   https://github.com/Ajay-Dangodara/Stock-Price-Forecast.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Example
Enter a stock ticker like AAPL, select a date range, and click Predict to see:

- Actual vs Predicted chart

- Next 30 days forecast

- Model accuracy and error metrics

## 🧠 Tech Stack
- Streamlit

- TensorFlow / Keras

- GRU (Recurrent Neural Network)

- Scikit-learn

- Plotly

- yFinance

## 📝 License
This project is open-source and available under the MIT License.