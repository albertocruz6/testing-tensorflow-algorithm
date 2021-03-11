import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()  #
data = scale(digits.data)  # scale between -1 to 1

y = digits.target

# number of clusters

# k = len(np.unique(y)) Dynamic way
k = 10
sample, features = data.shape

'''
Method to measure accuracy of clusters
'''


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


'''
n_cluster = number of clusters
init = where clusters will start
n_init = number of iterations of centroids
max_iter = max amount of changes that centroids can have
'''
clf = KMeans(n_clusters=k, init="random", n_init=10, max_iter=100)
bench_k_means(clf, "1", data)
