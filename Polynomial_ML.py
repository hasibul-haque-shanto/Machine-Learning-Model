# Polynomial regression 
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
# To check if there is any missing values
dataset.isna().sum()
# filling the missing values with most frequent value
dataset['Familienstand'].fillna(dataset['Familienstand'].value_counts().index[0], inplace = True)
X = dataset[['Alter', 'Geschlecht','Familienstand','Kinder', 'AndereHausstandsmitglieder', 'LohnWoche','Arbeitszeitkategorie', 'Arbeitszeitwoche', 'ArbeitstageWoche', 'Erstreserve' ]]
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
X[:,8] = labelencoder_X.fit_transform(X[:,8])

# Feature scalling
from sklearn import preprocessing
maxab_X = preprocessing.MaxAbsScaler()
X = maxab_X.fit_transform(X)
maxab_Y = preprocessing.MaxAbsScaler()
Y = maxab_Y.fit_transform(Y)

# Fitting Linear Regression to training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Fitting Polynomial Regression to training set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the polynomial regression
#plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting a new reslut using polynomial regression

Y_pred = lin_reg_2.predict(X_poly)

from sklearn.metrics import mean_squared_error
import math
MSE = mean_squared_error(Y, Y_pred)
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)