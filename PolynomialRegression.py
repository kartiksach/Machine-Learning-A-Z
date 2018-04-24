"""
Authour @ kartiksach
Platform @ Kartik's Macbook Air

This file conatins the entire polynomial regression code
as covered in Machine Learning A-Z
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('PolynomialRegression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Implementing Linear regression
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

# Visualising Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'blue')
plt.title('Linear Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Implementing Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 3)
X_poly = polyReg.fit_transform(X)
linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# Visualising Polynomial Regression
# X_grid is used for smoothening the curve with step = 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linReg2.predict(polyReg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Predicting Salary via Linear Regression Model
y_pred = linReg.predict(6.5)

# Predicting Salary via Polynomial Regression Model
y_pred_poly = linReg2.predict(polyReg.fit_transform(6.5))