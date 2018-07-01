#------------------------------------------------------------------------------#
# Chapter 4: Data Pre-Processing                                               #
#------------------------------------------------------------------------------#

import sbs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# skikit-learn preprocessing tools
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel


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
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# page 109 - more on creating good splits for test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#------------------------------------------------------------------------------#
# Scaling Features                                                             #
#------------------------------------------------------------------------------#

# normalization
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# standardization
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#------------------------------------------------------------------------------#
# Regularization                                                               #
#------------------------------------------------------------------------------#

# use OvR by default
lr = LogisticRegression(penalty="l1", C=0.1)
lr.fit(X_train_std, y_train)

# 98% on both does not indicate overfitting
#print("Training accuracy:", lr.score(X_train_std, y_train))
#print("Test accuracy:", lr.score(X_test_std, y_test))

# multiclass data (3 grapes)
# class 1 versus 2 and 3
# class 2 versus 1 and 3
# class 3 versus 1 and 2
#print(lr.intercept_)

# 3 rows of weight coefs for each, with 13 weights from the 13 features
# sparse data, lots of 0s
# l1 reg. serves for feature selection to train a model that is robust to the
# potentially irrelevant features in the dataset
#print(lr.coef_)

# regularization paths
# weight coefs of the different features for different regularization strengths
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "pink",
          "lightgreen", "lightblue", "gray", "indigo", "orange"]
weights, params = [], []
for c in np.arange(-4.0,6.0):
    lr = LogisticRegression(penalty="l1", C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights += [lr.coef_[1]]
    params += [10**c]

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)
plt.axhline(0, color="black", linestyle="--", linewidth=3)
plt.xlim([10**(-5), 10**5])

plt.ylabel("weight coefficient")
plt.xlabel("C")
plt.xscale("log")
plt.legend("upper left")
ax.legend(loc="upper center", bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.title("Regularization paths")
plt.tight_layout()
plt.show()              

#------------------------------------------------------------------------------#
# Sequential Backward Selection (SBS) with KNN model                           #
#------------------------------------------------------------------------------#

knn = KNeighborsClassifier(n_neighbors=2)
sbs = sbs.SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker="o")
plt.ylim([0.7,1.1])
plt.ylabel("Accuracy")
plt.xlabel("Number of Features")
plt.grid()
plt.title("K Neighbors: Feature Selection with SBS")
plt.show()

# subsets_[0] is 13 on plot, subsets_[11] is 1 on plot
# belos the fifth point on the fifth point on the plot with 5 features
k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

# evaluate knn on the original test set
# lower test accuracy here shows slight overfitting
knn.fit(X_train_std, y_train)
print("Orginal Feature Set")
print("Training accuracy:", knn.score(X_train_std, y_train))
print("Test accuracy:", knn.score(X_test_std, y_test))

# smaller overfitting gap, improved accuracy on test set
knn.fit(X_train_std[:, k5], y_train)
print("Selected Feature Set")
print("Training accuracy:", knn.score(X_train_std[:, k5], y_train))
print("Test accuracy:", knn.score(X_test_std[:, k5], y_test))
print("")
# Resource for more feature selection algorithms:
# there are a lot of good feature selecting algorithms in scikit-learn 
# https://scikit-learn.org/stable/modules/feature_selection.html

#------------------------------------------------------------------------------#
# Feature Importance with Random Forests                                       #
#------------------------------------------------------------------------------#

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Random Forests: Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], color="lightblue", align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# color intensity of the wine is the most discriminative feature in the dataset 
# based on the average impurity decrease in th 10,000 decision trees

# simple way to extract the top features:
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:', X_selected.shape[1])
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))




