# Linear Regression Model Documentation: Predicting Nigeria's Inflation Rate
This documentation outlines the process of developing and evaluating a linear regression model to predict Nigeria's inflation rate using real-world economic data. The model leverages the relationship between various Consumer Price Index (CPI) components and key economic indicators to predict the inflation rate, and its performance is measured using the Mean Absolute Error (MAE) metric.
________________________________________
# Problem Statement
•	Objective:
To create a predictive model that can estimate Nigeria's inflation rate based on select features of various CPI components.
•	Goal:
The model will take input values for selected economic features (like CPI components) and predict the inflation rate, providing a tool for forecasting and understanding economic trends.
________________________________________
# Data Description
The dataset provides a comprehensive overview of Nigeria's monthly inflation rates from March 2003 to June 2024. It also includes key economic indicators such as:
•	Crude oil prices
•	Production levels
•	Various Consumer Price Index (CPI) components
This dataset is ideal for time series analysis, forecasting, and economic modeling, capturing significant trends that are valuable for economic prediction.
________________________________________
## Model Training
Import Statements and Dataset
To begin with, the necessary libraries and modules are imported, and the dataset is loaded into the model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
## Data Splitting
The dataset contains both features (independent variables) and the target variable (inflation rate). Here, we use one feature, CPI_Food, for model training.
•	X (features): Consumer Price Index (CPI) for food
•	y (target): Inflation rate
The data is split into training and test sets, with an 80%/20% ratio for training and testing, respectively:

X = data['CPI_Food']
y = data['Inflation Rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Model Training
The linear regression model is trained using the training set (X_train and y_train):

model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)  # Reshaping X_train for fitting the model
________________________________________


# Model Evaluation
After training the model, predictions are made on both the training and test sets. The performance is evaluated using the Mean Absolute Error (MAE) metric, which measures the average magnitude of errors in a set of predictions. A lower MAE indicates better model accuracy.
•	Baseline Model MAE: 3.45 (This is the performance of a simple guess, like predicting the average value of the target variable for all data points.)
•	Model (Train) MAE: 2.42 (The MAE on the training data, showing how well the model fits the training set.)
•	Model (Test) MAE: 2.33 (The MAE on the test data, showing how well the model generalizes to unseen data.)
The test MAE of 2.33 demonstrates that the model’s predictions are closer to the actual inflation rates compared to the baseline model (MAE of 3.45), indicating an improvement in predictive accuracy.
python
Copy
# Make predictions on train and test sets
train_predictions = model.predict(X_train.values.reshape(-1, 1))
test_predictions = model.predict(X_test.values.reshape(-1, 1))

# Calculate MAE for training and testing sets
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
________________________________________

# Conclusion
The model has demonstrated a significant improvement in predictive accuracy compared to a baseline model, achieving a test MAE of 2.33. This indicates that the linear regression model, trained on CPI data (in this case, CPI_Food), is a useful tool for predicting Nigeria's inflation rate.
Further improvements could involve:
•	Incorporating additional features (e.g., crude oil prices, production levels) into the model
•	Evaluating more advanced models such as polynomial regression or machine learning models (e.g., Random Forest or XGBoost)
________________________________________
# Future Work
Future work can focus on:
•	Expanding the feature set to include more economic indicators.
•	Evaluating model performance over longer time periods or using different time series techniques for more accurate forecasting.
•	Experimenting with advanced models to improve prediction accuracy and handle non-linearity in the data.

