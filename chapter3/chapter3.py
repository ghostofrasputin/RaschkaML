#------------------------------------------------------------------------------#
# Chapter 3: Machine Learning Classifiers using Scikit-Learn                   #
#------------------------------------------------------------------------------#

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# sklearn 
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split # deprecated

# sklearn models
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#------------------------------------------------------------------------------#
# Perceptron with skikit-learn                                                 #
#------------------------------------------------------------------------------#

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
plt.title("Perceptron")
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
plt.title("Sigmoid Function Example")
plt.show()

#------------------------------------------------------------------------------#
# Logistic Regression with scikit-learn                                        #
#------------------------------------------------------------------------------#

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.title("Logistic Regression")
plt.show()

#------------------------------------------------------------------------------#
# L2 regularization for Logistic Regression                                    #
#   L2 reg. path of the two weight coefficents                                 #
#------------------------------------------------------------------------------#

# Note: random_state paramter is a seed to ensure repeatable results

# weight coeffcients shrink if the paramter C is decreased, thus increasing
# the regularization strength

weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.0**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights += [lr.coef_[1]]
    params += [10.0**c]

weights = np.array(weights)
plt.plot(params, weights[:, 0], label="petal length")
plt.plot(params, weights[:, 1], linestyle="--", label="petal width")
plt.ylabel("weight coeffcients")
plt.xlabel("C")
plt.legend(loc="upper left")
plt.xscale("log")
plt.title("L2 Regularization for LR")
plt.show()

#------------------------------------------------------------------------------#
# Support Vector Machines (SVM) with scikit-learn                              #
#------------------------------------------------------------------------------#

svm = SVC(kernel="linear", C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title("Support Vector Machine")
plt.show()

# Scikit-learn offers stochastic gradient descent variations of the perceptron,
# logistic regression, and support vector machines in the SGD Classifier class.
#
# from sklearn.linear_model import SGDClassifier
# ppn = SGDClassifier(loss="perceptron")
# lr = SGDClassifier(loss="log")
# svm = SGDClassifier(loss="hinge")

np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c="b", marker="x", label="1")
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c="r", marker="s", label="-1")
plt.ylim(-3.0)
plt.legend()
plt.tight_layout()
plt.title("Random Xor Data")
plt.show()


# gamma can be understood as a cut-off paramter for the Gaussian sphere
# if we increase the value, we increase the influence or reach of the training 
# samples, which leads to a softer decision boundary.
svm = SVC(kernel="rbf", random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc="upper left")
plt.title("Support Vector Machine for Xor data. 0.1 gamma")
plt.show()

# small gamma creates a soft decision boundary of the RBF kernel SVM model
svm = SVC(kernel="rbf", random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title("Support Vector Machine for Iris data. 0.2 gamma")
plt.show()

# although the model fits the training data very well, a classfier like this
# will likely have a high generalization error on unseen data, so gamma
# plays an important role in controlling overfitting
svm = SVC(kernel="rbf", random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title("Support Vector Machine for Iris data. 100 gamma")
plt.show()

#------------------------------------------------------------------------------#
# Decision Trees with scikit-learn                                             #
#------------------------------------------------------------------------------#

# visual comparison of the 3 impurity criteria

def gini(p):
    return p*(1-p) + (1-p)*(1-(1-p))

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2((1-p)) 
    
def error(p):
    return 1 - np.max([p, 1-p])
    
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p!=0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)

a = [ent, sc_ent, gini(x), err]
b = ["Entropy", "Entropy (scaled)", "Gini Impurity", "Misclassification Error"]
c = ["-", "-", "--", "-."]
d = ["black", "lightgray", "red", "green", "cyan"] 
for i, lab, ls, c in zip(a,b,c,d):
    line = ax.plot(x, i, label=lab, linestyle=ls, color=c)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color="k", linestyle="--")
ax.axhline(y=1.0, linewidth=1, color="k", linestyle="--")
plt.ylim([0,1.1])
plt.xlabel("p(i=1)")
plt.ylabel("Impurity Index")
plt.title("Impurity Comparison")
plt.show()
    
# Decision Tree
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.title("Decision Tree")
plt.show()


#------------------------------------------------------------------------------#
# Random Forests with scikit-learn                                             #
#------------------------------------------------------------------------------#

# Decision Tree
# uses 10 decision trees via n_estimators
# entropy criterion as impurity measure to split the nodes
# n_jobs parallize the model training using multiple cores of our computer
#   ^ isn't necassry because of the small training set and small number of trees
forest = RandomForestClassifier(criterion="entropy", n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.title("Random Forest")
plt.show()

#------------------------------------------------------------------------------#
# K-Nearest Neighbors scikit-learn                                             #
#------------------------------------------------------------------------------#

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standarized]')
plt.legend(loc='upper left')
plt.title("K-Nearest Neighbors")
plt.show()








