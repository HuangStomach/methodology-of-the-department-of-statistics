import numpy as np
from copy import deepcopy
from typing import TypeVar

# k均值聚类算法
P = TypeVar('P', bound='KMeansClustering')
class KMeansClustering:
    def __init__(self, k = 2) -> None:
        self._k = k
        pass

    def _distance(self, x1, x2) -> float:
        # 欧式距离
        total = 0.0
        for i in range(self._m):
            total += abs(x1[i] - x2[i]) ** 2
        return total
    
    def _rest_center(self, k) -> P:
        for m in range(self._m):
            total = 0.0
            for x in self._classify[k]:
                total += x[m]
            self._centers[k][m] = total / len(self._classify[k]) # 更新簇中心每个维度的数值
        return self

    def fit(self, X) -> P:
        self._n = len(X)
        self._m = len(X[0])
        self._X = deepcopy(X)
        self._y = np.zeros(self._n)

        # 取出前k个作为簇的中心
        self._centers = []
        self._classify = []
        for k in range(self._k):
            self._centers.append(self._X[k].copy())
            self._classify.append([])

        while True: # 只要修改过样本的分类就重新进行计算
            is_modified = False
            for i in range(self._n):
                dis = []
                # 将各簇中心和样本进行距离计算，将样本分类至距离最近的簇
                for k in range(self._k):
                    dis.append(self._distance(self._X[i], self._centers[k]))
                
                new_cluster = np.argmin(dis)
                self._classify[new_cluster].append(self._X[i]) # 空间换时间，将此次分类中同一分类的样本点放置在一个集合中
                if new_cluster != self._y[i]: # 更新样本的分类
                    is_modified = True
                    self._y[i] = new_cluster
            
            if is_modified == False: break

            # 重新计算簇中心
            for k in range(self._k):
                self._rest_center(k)
                self._classify[k] = []

        return self

    def score(self, y) -> float:
        _revise = []
        pass

from sklearn.datasets import load_boston
boston_dataset = load_boston()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    boston_dataset['data'], boston_dataset['target'], random_state=5
)

cluster = KMeansClustering(2)
cluster.fit(X_train)
