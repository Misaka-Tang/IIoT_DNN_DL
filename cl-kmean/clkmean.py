import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


class ClKmeans:
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.tree = DecisionTreeClassifier()

    def fit(self, X):
        self.kmeans.fit(X)
        self.tree.fit(X, self.kmeans.labels_)

    def predict(self, X):
        return self.tree.predict(X)
    
