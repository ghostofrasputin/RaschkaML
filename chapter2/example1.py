# Training a perceptron model on an iris data set

import numpy as np
import pandas as pd
import perceptron as p
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# print out last five lines to test if iris data
# was loaded correctly
df =  pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data', header=None)
# print(df.tail())

# extract first 100 labels and convert labels into ints
# Sertosa - -1
# Versicolor - 1 
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# first column - sepal length
# third column - petal length
# assign these values to feature matrix X
X = df.iloc[0:100, [0,2]].values
# print(X)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc="upper left")
plt.show()

# train Perceptron on Iris dataset extracted (y)
# Perceptron converges after the 6th epoch
ppn = p.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, color='red', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.show()

# convenience function to visualize the decision boundaries for 2D datasets
def plot_decision_regions(X, y, classifier, res=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # min and max of sepal length
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # min and max of petal length
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max, res),np.arange(x2_min, x2_max, res))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], 
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=c1) 

plot_decision_regions(X, y, ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()                    




                 