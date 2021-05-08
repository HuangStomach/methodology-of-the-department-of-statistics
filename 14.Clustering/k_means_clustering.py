import numpy as np
from copy import deepcopy
from typing import TypeVar

# k均值聚类算法
P = TypeVar('P', bound='KMeansClustering')
class KMeansClustering:
    def __init__(self) -> None:
        pass

    def _distance(self, x1, x2) -> float:
        # 欧式距离
        total = 0.0
        for i in range(self._m):
            total += abs(x1[i] - x2[i]) ** 2
        return total ** 0.5

    def fit(self, X) -> P:
        self._X = deepcopy(X)
        self._n = len(X)
        self._m = len(X[0])

        return self


from sklearn.datasets import load_iris
iris_dataset = load_iris()
X = iris_dataset['data'][:5]

cluster = KMeansClustering()
cluster.fit(X)
