import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Support Vector Machine!
# Load local data set
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# print(x_train, y_train)
classes = ['malignant', 'benign']

# kernels:  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# poly is better but requires more computational power (can modify with degree=3,4,5,)
# C = soft margin (default 1)
clf = svm.SVC(kernel="poly", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test,y_pred)
print(acc)