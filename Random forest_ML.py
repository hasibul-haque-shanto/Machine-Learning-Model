# Random Forest Classification
# Importing the libraries
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
dataset.loc[dataset['Endkosten'] >25000, 'Anspruchsart'] = 1
dataset.loc[dataset['Endkosten'] <=25000, 'Anspruchsart'] = 0 
Y = dataset['Anspruchsart']

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

#import Library
from imblearn.over_sampling import SMOTE
smote = SMOTE()
# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X, Y)

# Splitting dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_smote, y_smote, test_size = 0.25, random_state = 36484)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 170, criterion = 'entropy', random_state = 36484)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Evaluating the model 
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,Y_pred ))
print(confusion_matrix(Y_test,Y_pred))
