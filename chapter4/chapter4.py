#------------------------------------------------------------------------------#
# Chapter 4: Data Pre-Processing                                               #
#------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------#
# Handling Missing Numerical Data                                              #
#------------------------------------------------------------------------------#

# build data frame from comma-seperated values (CSV) file 
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''

# StringIO reads string as if it were a regular CSV file on the hard drive
df = pd.read_csv(StringIO(csv_data))
#print(df)

# isnull returns DF of booleans
# sum returns number of null values in each column
#print(df.isnull().sum())        

# Scikit-Learn was developed to use numpy arrays, but sometimes using pandas'
# DataFrame is easier for pre-processing data.
#
# With DataFrames you can always access the numpy array with the values
# attributes:
#print(df.values)

# Feature columns or sample rows can be simply removed from 
# the dataset entirely.
# Drops rows 2 and 3
#print(df.dropna())
# drops columns 3 and 4
#print(df.dropna(axis=1))
# only drop rows where all columns are Nan, does nothing in this example
#print(df.dropna(how="all"))
# only drop rows that have not at last 4 non-Nan values
#print(df.dropna(thresh=4))
# only drop rows where NaN appear in specific columnns,such as C
#print(df.dropna(subset=["C"]))

# imputer to fill in NaN values with mean value
# axis = 0 calculates column means
# axis = 1 calculates row means
# strategy can also use median or most_frequent (mode) 
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr.fit(df)
imputed_data = imr.transform(df.values)
#print(imputed_data)

#------------------------------------------------------------------------------#
# Handling Categorical Data                                                    #
#------------------------------------------------------------------------------#

df = pd.DataFrame([["green", "M", 10.1, "class1"],
                   ["red", "L", 13.5, "class2"],
                   ["blue", "XL", 15.3, "class1"]])
df.columns = ["color", "size", "price", "classlabel"]
#print(df)

size_mapping = {"XL":3, "L":2, "M":1}
df["size"] = df["size"].map(size_mapping)
#print(df)

# creating a inverted map if values ever need to be set back 
inv_size_mapping = {v:k for k, v in size_mapping.items()}

#print(np.unique(df["classlabel"])) # ['class1' 'class2']
# unique creates array of the 2 unique elements and each label
# is given a number
class_mapping = {label:idx for idx, label in enumerate(np.unique(df["classlabel"]))}
#print(class_mapping)

df["classlabel"] = df["classlabel"].map(class_mapping)
#print(df)

# creating a inverted map if values ever need to be set back 
inv_class_mapping = {v:k for k, v in class_mapping.items()}

# alternative method of class label integer mapping with scikit-learn
#class_le = LabelEncoder()
#y = class_le.fit_transform(df["classlabel"].values)
#print(y)
#y = class_le.inverse_transform(y)
#print(y)

#X = df[["color", "size", "price"]].values
#color_le = LabelEncoder()
#X[:, 0] = color_le.fit_transform(X[:, 0])
#print(X)
#ohe = OneHotEncoder(categorical_features=[0])
#ohe.fit_transform(X).toarray()

# conveinent way to create dummy features
# only converts string columns and leaves everything else alone
print(pd.get_dummies(df[["color", "size", "price"]]))

#------------------------------------------------------------------------------#
# Partitioning a dataset in training and test sets                             #
#------------------------------------------------------------------------------#

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

#print('Class labels', np.unique(df_wine['Class label']))
#print(df_wine.head())

# skip class label column
X, y = df_wine.iloc[:, 1:].values, df.wine.iloc[:, 0].values

# page 109 - more on creating good splits for test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


