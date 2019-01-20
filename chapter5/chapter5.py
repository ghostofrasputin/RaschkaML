#------------------------------------------------------------------------------#
# Chapter 5: Compressing Data via Dimensionality Reduction                     #
#------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib
from matplotlib.colors import ListedColormap

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

#------------------------------------------------------------------------------#
# Total and Explained Variance                                                 #
#------------------------------------------------------------------------------#

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# Splitting the data into 70% training and 30% test subsets.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,random_state=0)

# Standardizing the data.
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

#------------------------------------------------------------------------------#
# Feature Transformation                                                       #
#------------------------------------------------------------------------------#

# sort the eigenpairs by decreasing order of the eigenvalues
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# select the 2 eigen vectors that correspond with the 2 largest values
# to capture 60% of the variance in the data set
# note: In practice, the number of principal components has to be determined
# by the trade-off between computational efficiency and the performance of the
# classifier
feature1 = eigen_pairs[0][1][:, np.newaxis]
feature2 = eigen_pairs[1][1][:, np.newaxis]
#print(feature1)
#print(feature2)
w = np.hstack((feature1, feature2))
print("Matrix W\n",w)

# W is now a 13x2 dimensional projection matrix from the top 2 eigen vectors
# With sample x represented as 1x13 row vector and PCA subspace obtaining x'
# a 2 dimensional sample vector sonsisting of 2 new features:
# x' = xW

#print(X_train_std[0].dot(w))

# We can transform the entire 124x13 training dataset onto the 2 principal
# components by calculating the matrix dot product:
# X' = XW

X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for cl, c, m in zip(np.unique(y_train), colors, markers):
    f1_x_axis = X_train_pca[y_train == cl, 0]
    f2_y_axis = X_train_pca[y_train == cl, 1]
    plt.scatter(f1_x_axis, f2_y_axis, c=c, label=cl, marker=m)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.show()

# From the plot, we can see the data is more spread along the x-axis for the
# 1st principal component than the 2nd principal component (y-axis), which
# shows consistency with the explained variance ratio plot
#
# From this, we can intuitively see that a linear classifier will likely be able
# to seperate the classes well.     


#------------------------------------------------------------------------------#
# Principal Component Analysis with sckit-learn                                #
#------------------------------------------------------------------------------#

# sklearn method of copying the bar graph from the first section
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

# sklearn method to scatter data from the first section
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

def plot_decision_regions(X, y, classifier, test_idx=None, res=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl, in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label = cl, edgecolor="black")

# example using the scikit-learn PCA module

# train data set
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# pca and lr already fit
X_test_pca = pca.transform(X_test_std)
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

# get all explaind variance ratio
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)
