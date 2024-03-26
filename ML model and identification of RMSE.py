# Exercise-2

# Import Libraries
import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv('train.csv')
# To check if there is any missing values
dataset.isna().sum()
# filling the missing values with most frequent value
dataset['Familienstand'].fillna(dataset['Familienstand'].value_counts().index[0], inplace = True)
# definning independent and dependent variables
X = dataset[['Alter', 'Geschlecht', 'Familienstand', 'Kinder', 'AndereHausstandsmitglieder', 'LohnWoche', 'Arbeitszeitkategorie', 'Arbeitszeitwoche', 'ArbeitstageWoche', 'Erstreserve' ]]
# taking target variable in log scale
Y = np.log1p(dataset['Endkosten'])
Y = Y.to_frame(name="Endkosten")

# Taking Care of Categorical data
# Geschlecht
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X['Geschlecht'] = labelencoder_X.fit_transform(X['Geschlecht'])
onehotencoder = OneHotEncoder()
X_1 = onehotencoder.fit_transform(X[['Geschlecht']]).toarray()
X = np.append(X_1, X, axis =1)
# Avoiding the Dummy Variable Trap
X = np.delete(X,4,1)
X = np.delete(X,0,1)

#Familienstand
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder()
X_2 = onehotencoder.fit_transform(X[:,[3]]).toarray()
X = np.append(X_2, X, axis =1)
# Avoiding the Dummy Variable Trap
X = np.delete(X,7,1)
X = np.delete(X,0,1)

# Arbeitszeitkategorie
X[:,8] = labelencoder_X.fit_transform(X[:,8])

# Feature scaling
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

# Evaluating the model
from sklearn.metrics import mean_squared_error
import math
MSE = mean_squared_error(Y, Y_pred)
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)