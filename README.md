
![Logo](https://github.com/dasdebanna/CineGenius-Revolutionizing-Movie-Recommendations/blob/main/images/movie-recommendation.png)

# Stock Price Prediction with Random Forest & Live Yahoo Data

This project combines the techniques of Stock Price Prediction using Random Forest Regression and Live Data Fetching from Yahoo Finance to create a comprehensive solution for predicting stock prices. We will leverage the power of the ```yfinance``` library to fetch real-time stock data and the Random Forest Regression algorithm to build a predictive model.

The key components of this project are:

1. Fetching historical stock data using the ```yfinance``` library.
2. Preprocessing the data and preparing it for the machine learning model.
3. Performing exploratory data analysis on the stock data.
4. Building and training the Random Forest Regression model.
5. Evaluating the model's performance.
6. Predicting the future stock price using the trained model.

## Data Preprocessing
We start by importing the necessary libraries and defining the ticker symbol, start date, and end date for the stock data we want to fetch.

```python
import yfinance as yf
import pandas as pd
import os

ticker = "DLF.NS"
start_date = "2021-01-04"
end_date = "2024-05-01"

data = yf.download(ticker, start=start_date, end=end_date)
df = pd.DataFrame(data)
```
Next, we perform some basic data preprocessing tasks, such as converting the index to a datetime object and dropping unnecessary columns.

```python
df['date'] = pd.to_datetime(df.index)
df.drop(['date', 'Volume'], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
```


## Exploratory Data Analysis
To better understand the stock data, we can create a candlestick chart using Plotly.

```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'])])

fig.update_layout(title='Stock Price Chart DLF.NS', yaxis_title='Price (₹)', xaxis_rangeslider_visible=False)
fig.show()
```
![App Screenshot](https://github.com/dasdebanna/CineGenius-Revolutionizing-Movie-Recommendations/blob/main/images/screenshot-1.png)

The chart provides a visual representation of the stock's open, high, low, and close prices over time.
## Model Building
We will use the Random Forest Regression algorithm to predict the stock price. We split the dataset into training and testing sets, and then train the model on the training data.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df[['Open', 'Close', 'High', 'Low', 'Adj Close']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=79)

rf = RandomForestRegressor(n_estimators=200, random_state=79)
rf.fit(X_train, y_train)
```

## Model Evaluation
To evaluate the model's performance, we calculate the Mean Squared Error (MSE) on the testing set. The Mean Squared Error is a measure of the average squared difference between the predicted values and the actual values. A lower MSE indicates a better model fit.

```python
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
```
If the output of the Mean Squared Error is less than 1, it indicates that the model is performing well and the predictions are accurate. However, if the MSE is greater than 1, it means the model needs to be improved. In this case, you may want to try the following:

1. Explore additional features that could improve the model's predictive power.
2. Adjust the hyperparameters of the Random Forest Regressor, such as the number of estimators or the maximum depth of the trees.
3. Consider using a different machine learning algorithm or an ensemble method to see if it can outperform the Random Forest Regression model.
## Predicting Future Stock Price
Finally, we can use the trained model to predict the future stock price. In this example, we provide a new set of input values and ask the model to predict the corresponding stock price.
```python
import numpy as np

new_data = np.array([[888.799998, 907.500000, 881.900024, 891.849976, 891.849976]])
predicted_price = rf.predict(new_data)
print('Predicted Stock Price:', predicted_price[0])
```

## Conclusion
In this project, we have combined the techniques of Stock Price Prediction using Random Forest Regression and Live Data Fetching from Yahoo Finance to create a comprehensive solution for predicting stock prices. We have demonstrated the process of fetching historical stock data, preprocessing the data, building and training the Random Forest Regression model, evaluating the model's performance, and using the trained model to predict future stock prices.

This project can be further extended by incorporating additional features, such as market indicators, economic data, or news sentiment, to improve the model's predictive power. Additionally, you can experiment with other machine learning algorithms or ensemble methods to see if they can outperform the Random Forest Regression model.

Please feel free to explore the code, modify it, and use it as a starting point for your own stock price prediction projects.