#------------------------------------------------------------------------------#
# Training a perceptron model on an iris data set                              #
#------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import perceptron as p
import adaline_gd as agd
import adaline_sgd as asgd
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


#------------------------------------------------------------------------------#
# Training gradient descent adaline model on the iris dataset                  #
#------------------------------------------------------------------------------#

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada1 = agd.AdalineGD(eta=0.01, n_iter=10).fit(X,y)
ada2 = agd.AdalineGD(eta=0.0001, n_iter=10).fit(X,y)
for i in range(2):
    ada0 = ada1 if i==0 else ada2
    cost = ada0.cost_
    ax[i].plot(range(1, len(cost) + 1), np.log10(cost), marker = 'o')
    ax[i].set_xlabel('Epochs')
    ax[i].set_ylabel('Sum Squared Error')
    title = "Eta - 0.01" if i==0 else "Eta - 0.0001"
    ax[i].set_title(title)
plt.show()

# left plot:
# ada 1 (eta - 0.01) shows us what happens when we get a learning rate 
# that is too big. instead of minimzing the CF, the error becomes larger every 
# epoch since it overshot the global minimum. 
# 
# right plot:
# ada 2 (eta - 0.0001) is too small so even though the cost decreases it would
# take a large number of epochs to converge

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada3 = agd.AdalineGD(eta=0.01, n_iter=15).fit(X_std,y)
plot_decision_regions(X_std, y, ada3)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]') 
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada3.cost_)+1), ada3.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum Squared Error')
plt.show()
            

#------------------------------------------------------------------------------#
# Training stochastic gradient descent adaline model on the iris dataset       #
#------------------------------------------------------------------------------#

ada4 = asgd.AdalineSGD(eta=0.01, n_iter=15, random_state=1).fit(X_std, y)
plot_decision_regions(X_std, y, ada4)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]') 
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada4.cost_)+1), ada4.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()



                 