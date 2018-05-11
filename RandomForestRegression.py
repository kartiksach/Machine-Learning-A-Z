"""
Author @ kartiksach
Platform @ Kartik's Macbook Air

This file contains the entire Random Forest regression code
as covered in Machine Learning A-Z
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('RandomForestRegression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Implementing Random Forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

# Predicting salary
y_pred = regressor.predict(6.5)

# Visualising regression model
# X_grid is used for higher resolution ie smoother curves
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Random Forest Regression')
plt.show()