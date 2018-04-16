"""
Authour @ kartiksach
Platform @ Kartik's Macbook Air

This file conatins the entire data pre-prepocessing code 
as covered in Machine Learning A-Z
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding values for categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
labelEncodery = LabelEncoder()
X[:, 0] = labelEncoderX.fit_transform(X[:, 0])
y = labelEncodery.fit_transform(y)
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

# Splitting dataset for training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Perform feature scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)