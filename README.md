# Linear Regression Model Documentation: Predicting Nigeria's Inflation Rate

This documentation outlines the process of developing and evaluating a linear regression model to predict Nigeria's inflation rate using real-world economic data. The model leverages the relationship between various Consumer Price Index (CPI) components and key economic indicators to predict the inflation rate, and its performance is measured using the Mean Absolute Error (MAE) metric.
________________________________________
# Problem Statement

•	**Objective:** <br>
To create a predictive model that can estimate Nigeria's inflation rate based on select features of various CPI components. <br>

•	**Goal:** <br>
The model will take input values for selected economic features (like CPI components) and predict the inflation rate, providing a tool for forecasting and understanding economic trends.
________________________________________
# Data Description

The dataset provides a comprehensive overview of Nigeria's monthly inflation rates from March 2003 to June 2024. It also includes key economic indicators such as:<br>
•	Crude oil prices<br>
•	Production levels<br>
•	Various Consumer Price Index (CPI) components<br>

This dataset is ideal for time series analysis, forecasting, and economic modeling, capturing significant trends that are valuable for economic prediction.
________________________________________
## Model Training

**Import Statements and Dataset**  <br>
To begin with, the necessary libraries and modules are imported, and the dataset is loaded into the model.

import pandas as pd <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>
import seaborn as sns <br>
import sklearn <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.linear_model import LinearRegression <br>
from sklearn.metrics import mean_absolute_error <br>
from sklearn.metrics import mean_squared_error, r2_score <br>

## Data Splitting

The dataset contains both features (independent variables) and the target variable (inflation rate). Here, we use one feature, CPI_Food, for model training.<br>
•	X (features): Consumer Price Index (CPI) for food <br>
•	y (target): Inflation rate <br>
The data is split into training and test sets, with an 80%/20% ratio for training and testing, respectively: <br>

X = data['CPI_Food'] <br>
y = data['Inflation Rate'] <br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) <br>

## Model Training

The linear regression model is trained using the training set (X_train and y_train): <br>

model = LinearRegression() <br>
model.fit(X_train.values.reshape(-1, 1), y_train)  # Reshaping X_train for fitting the model
________________________________________


# Model Evaluation

After training the model, predictions are made on both the training and test sets. The performance is evaluated using the Mean Absolute Error (MAE) metric, which measures the average magnitude of errors in a set of predictions. A lower MAE indicates better model accuracy. <br>

•	Baseline Model MAE: 3.45 (This is the performance of a simple guess, like predicting the average value of the target variable for all data points.) <br>
•	Model (Train) MAE: 2.42 (The MAE on the training data, showing how well the model fits the training set.) <br>
•	Model (Test) MAE: 2.33 (The MAE on the test data, showing how well the model generalizes to unseen data.) <br>
The test MAE of 2.33 demonstrates that the model’s predictions are closer to the actual inflation rates compared to the baseline model (MAE of 3.45), indicating an improvement in predictive accuracy.
python
Copy
# Make predictions on train and test sets

train_predictions = model.predict(X_train.values.reshape(-1, 1)) <br>
test_predictions = model.predict(X_test.values.reshape(-1, 1)) <br>

# Calculate MAE for training and testing sets

train_mae = mean_absolute_error(y_train, train_predictions) <br>
test_mae = mean_absolute_error(y_test, test_predictions) <br>
________________________________________

# Conclusion

The model has demonstrated a significant improvement in predictive accuracy compared to a baseline model, achieving a test MAE of 2.33. This indicates that the linear regression model, trained on CPI data (in this case, CPI_Food), is a useful tool for predicting Nigeria's inflation rate. <br>

Further improvements could involve:<br>
•	Incorporating additional features (e.g., crude oil prices, production levels) into the model <br>
•	Evaluating more advanced models such as polynomial regression or machine learning models (e.g., Random Forest or XGBoost) <br>
________________________________________

# Future Work

Future work can focus on: <br>
•	Expanding the feature set to include more economic indicators. <br>
•	Evaluating model performance over longer time periods or using different time series techniques for more accurate forecasting. <br>
•	Experimenting with advanced models to improve prediction accuracy and handle non-linearity in the data.

