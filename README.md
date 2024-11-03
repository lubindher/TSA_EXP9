# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 
### Name: Lubindher S
### Reg No: 212222240056
### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('NVIDIA_Stock_Price.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])  # Adjust format as per your data
data.set_index('Datetime', inplace=True)

# Plot the Power Consumption to inspect for trends
plt.figure(figsize=(10, 5))
plt.plot(data['Temperature'], label='NVIDIA_Stock_Price')
plt.title('Time Series of Power Consumption')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

# Check stationarity with ADF test
result = adfuller(data['NVIDIA_Stock_Price'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, apply differencing
data['NVIDIA_Stock_Price_diff'] = data['NVIDIA_Stock_Price'].diff().dropna()
result_diff = adfuller(data['NVIDIA_Stock_Price_diff'].dropna())
print('Differenced ADF Statistic:', result_diff[0])
print('Differenced p-value:', result_diff[1])

# Plot ACF and PACF for differenced data
plot_acf(data['NVIDIA_Stock_Price_diff'].dropna())
plt.title('ACF of Differenced Power Consumption')
plt.show()

plot_pacf(data['NVIDIA_Stock_Price_diff'].dropna())
plt.title('PACF of Differenced Power Consumption')
plt.show()

# Plot Differenced Representation
plt.figure(figsize=(10, 5))
plt.plot(data['NVIDIA_Stock_Price_diff'], label='Differenced Power Consumption', color='red')
plt.title('Differenced Representation of Power Consumption')
plt.xlabel('Date')
plt.ylabel('Differenced Power Consumption')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.legend()
plt.show()

# Use auto_arima to find the optimal (p, d, q) parameters
stepwise_model = auto_arima(data['NVIDIA_Stock_Price'], start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(stepwise_model.summary())

# Fit the ARIMA model using the optimal parameters
model = sm.tsa.ARIMA(data['NVIDIA_Stock_Price'], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 30 days (or adjust period based on your needs)
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['NVIDIA_Stock_Price'], label='Actual Power Consumption')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Power Consumption')
plt.title('ARIMA Forecast of Power Consumption')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['NVIDIA_Stock_Price']) - 1)
mae = mean_absolute_error(data['NVIDIA_Stock_Price'], predictions)
rmse = np.sqrt(mean_squared_error(data['NVIDIA_Stock_Price'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
```

### OUTPUT:

![download](https://github.com/user-attachments/assets/e4f96a01-6382-4261-9817-153cc68e25f6)

![324377856-acfca51e-71a8-4fbf-bea7-20332cb8b9b0](https://github.com/user-attachments/assets/7613b8ee-89ec-4905-9294-63a4a4ffa4da)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
