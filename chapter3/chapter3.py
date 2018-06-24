#------------------------------------------------------------------------------#
# Perceptron with skikit-learn                                                 #
#------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import train_test_split # deprecated

sc = StandardScaler()
iris = datasets.load_iris()
# petal length and petal width
X = iris.data[:, [2,3]]
# setosa 0, versicolor 1, virginica 2
y = iris.target

# 30/70 split
# 45 test samples and 105 training data samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# 4/45 samples misclassified:
y_pred = ppn.predict(X_test_std)
print("Misclassified sample %d" % (y_test != y_pred).sum()) # prints 4
# true vs predicted
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

def plot_decision_regions(X, y, classifier, test_idx=None, res=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, res), np.arange(x2_min, x2_max, res))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,  c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black',
                    alpha=1.0, linewidth=1, marker='o', s=100, label='test set')


# Training a perceptron model using the standardized training data:
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()

#------------------------------------------------------------------------------#
# Sigmoid function                                                             #
#------------------------------------------------------------------------------#

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------#
# Logistic Regression with scikit-learn                                        #
#------------------------------------------------------------------------------#

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel("petal length [standardized]")
plt.xlabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.show()




