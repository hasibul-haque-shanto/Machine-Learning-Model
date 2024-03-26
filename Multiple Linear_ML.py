# data preprocessing
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('train.csv')
X = dataset[['Alter', 'Geschlecht', 'Familienstand','Kinder', 'AndereHausstandsmitglieder', 'LohnWoche', 'Arbeitszeitkategorie', 'Arbeitszeitwoche', 'ArbeitstageWoche', 'Erstreserve' ]]
Y = np.log1p(dataset['Endkosten'])
Y = Y.to_frame(name="Endkosten")

# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X['Geschlecht'] = labelencoder_X.fit_transform(X['Geschlecht'])

onehotencoder = OneHotEncoder()
X_1 = onehotencoder.fit_transform(X[['Geschlecht']]).toarray()
X = np.append(X_1, X, axis =1)

# Avoiding the Dummy Variable Trap
X = np.delete(X,4,1)
X = np.delete(X,0,1)

#Family status
X[:,3] = labelencoder_X.fit_transform(X[:,3])

onehotencoder = OneHotEncoder()
X_2 = onehotencoder.fit_transform(X[:,[3]]).toarray()
X = np.append(X_2, X, axis =1)

# Avoiding the Dummy Variable Trap
X = np.delete(X,7,1)
X = np.delete(X,0,1)
# job category
X[:,9] = labelencoder_X.fit_transform(X[:,9])
onehotencoder = OneHotEncoder()
X_3 = onehotencoder.fit_transform(X[:,[9]]).toarray()
X = np.append(X_3, X, axis =1)

# Avoiding the Dummy Variable Trap
X = np.delete(X,11,1)
X = np.delete(X,0,1)

# Splitting dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scalling
from sklearn import preprocessing
maxab_X = preprocessing.MaxAbsScaler()
X_test = maxab_X.fit_transform(X_test)
maxab_Y = preprocessing.MaxAbsScaler()
Y_test = maxab_Y.fit_transform(Y_test)

# Fitting Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_test, Y_test)

# Predicting the test set result 
#Y_pred = regressor.predict(X_opt)

# Building the optimal model using Backwad Elimination
import statsmodels.api as sm
X_m = np.append(arr=np.ones((10800,1)).astype(int), values = X_test, axis = 1)
X_opt = X_m[:,[0,1,3,5,7,9,10]]
X_opt = np.array(X_opt, dtype=float)
regressor_ols = sm.OLS(Y_test, X_opt).fit()
print(regressor_ols.summary())

Y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
import math
MSE = mean_squared_error(Y_test, Y_pred)
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)