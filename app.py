import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define the stock ticker symbol and date range
start_date = "2010-01-01"
end_date = "2023-09-29"

st.title("Stock Price Prediction")
user_input = st.text_input("Enter Stock Ticker", "AAPL")

# Scraping the data from yahoo finance data website
df = yf.download(user_input, start=start_date, end=end_date)

# Add a slider for selecting the year
selected_year = st.slider(
    "Select Year",
    min_value=int(df.index.year.min()),
    max_value=int(df.index.year.max()),
    value=int(df.index.year.max()),
)

# Filter data for the selected year
df_selected_year = df[df.index.year == selected_year]

# Visualisation for the selected year
st.subheader(f"Closing Price vs Time chart for {selected_year}")
fig_selected_year = plt.figure(figsize=(12, 6))
plt.plot(
    df_selected_year.index, df_selected_year.Close
)  # Use df.index as x-axis values
plt.xlabel("Year")  # Add x-axis label
st.pyplot(fig_selected_year)

st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, "r")
plt.plot(ma200, "g")
plt.plot(df.Close, "b")
st.pyplot(fig)

# splitting data in training and testing
# first 70% of data used as training data for the model
data_training = pd.DataFrame(df["Close"][0 : int(len(df) * 0.70)])
# last 30% used for testing
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler

# scaled down training data using MinMaxScaler for the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# load my model
model = load_model("keras_model.h5")

# Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
# creating our xtest and ytest
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])


x_test = np.array(x_test)
y_test = np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_predicted)
# Calculate Mean Absolute Percentage Error (MAPE)
mape = (mae / np.mean(y_test)) * 100


# Define a range of years from 2010 to 2024
years = np.linspace(2010, 2024, len(y_test))

# final graph
st.subheader("Predictions vs Original")
# Display MAPE
st.write(f"Mean Absolute Percentage Error Between Predicted And Original: {mape:.2f}%")

fig2 = plt.figure(figsize=(12, 6))
plt.plot(years, y_test, "b", label="Original Price")
plt.plot(years, y_predicted, "r", label="Predicted Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
