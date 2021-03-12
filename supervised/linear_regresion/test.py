import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle  # Not sure for what it is
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # get list of attributes

# print(data.head())

predict = "G3"  # variable in data to predict (grade to come) LABEL

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

'''
Model creation & training

Subdivide the data into sections to avoid BIAS
If the complete data set was used to make the prediction it would always get 100% due to it knowing all of the variables
This doesnt happen in the real world

To avoid extreme bias we separate a chunk of data (10% in this case) as unknown data so it calculates with "real" 
conditions (unknown factors)
'''
best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

"""
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    # if accuracy improves save model
    if acc > best:
        '''
        Save model (use if has high accuracy and to avoid retraining)
        pickle lib
        '''
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# Reload model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

'''
Understanding output
'''
# slope (5 values mean five slopes)
print("Co:", linear.coef_)
# y intercept
print("Intercept:", linear.intercept_)
print()

'''
Predict
'''

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'absences'
style.use("ggplot")
plt.scatter(data[p], data[predict])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()