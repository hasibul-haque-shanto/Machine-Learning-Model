# SVR
# Regression Template

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('train.csv')
X = dataset[['Alter', 'Geschlecht','Kinder', 'AndereHausstandsmitglieder', 'LohnWoche', 'Arbeitszeitwoche', 'ArbeitstageWoche', 'Erstreserve' ]]
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

# Feature scalling
from sklearn import preprocessing
maxab_X = preprocessing.MaxAbsScaler()
X = maxab_X.fit_transform(X)
maxab_Y = preprocessing.MaxAbsScaler()
Y = maxab_Y.fit_transform(Y)

# Fitting the Regression to training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

regressor.fit(X, Y.ravel())

#predicting a new reslut using  regression
Y_pred = regressor.predict(X)


from sklearn.metrics import mean_squared_error
import math
MSE = mean_squared_error(Y, Y_pred)
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)
