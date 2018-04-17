"""
Authour @ kartiksach
Platform @ Kartik's Macbook Air

This file implements Linear Regression
as covered in Machine Learning A-Z
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing datset
dataset = pd.read_csv('LinearRegression.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Spliting dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Implementing linear regression on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting y based on x_test
y_pred = regressor.predict(X_test)

# Visulising linear regression
# Red points for training points
# Green points for testing points
# Blue line for regression model
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Expirience')
plt.xlabel('Expirience')
plt.ylabel('Salary')
plt.show()
