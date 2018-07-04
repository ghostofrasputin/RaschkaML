#------------------------------------------------------------------------------#
# Chapter 5: Compressing Data via Dimensionality Reduction                     #
#------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------#
# Total and Explained Variance                                                 #
#------------------------------------------------------------------------------#

# split data
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# standardize the data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# note: numpy.linalg.eigh has been implemented to decompose Hermetian matrices
# which is numerically more stable approach to work with symmetric matrices
# such as the covariance matrix, numpy.linalg.eigh always returns real 
# eigenvalues, whereas numpy.linalg.eig was designed to decompose nonsymmetric
# square matrices, so it may return complex eigenvalues in certain cases.
covariance_matrix = np.cov(X_train_std.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
#print("Eigenvalues",eigen_values)

# to reduce dimensionality of the dataset by compressing it onto a new fetaure
# space, we select the subset of eigenvectors (principal components) that
# contains the most of the information (variance)
total = sum(eigen_values)
# VER is simply the fraction of an eigenvalue over the total sum of the
# eigenvalues
var_exp = [(i/total) for i in sorted(eigen_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)
#
plt.bar(range(1,14), var_exp, alpha=0.5, align="center", label="individual explained variance")
plt.step(range(1,14), cum_var_exp, where="mid", label="cumulative explained variance")
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.legend(loc="best")
plt.show()
 

