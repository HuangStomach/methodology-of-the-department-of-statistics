import numpy as np
from typing import TypeVar

# 聚合聚类算法
P = TypeVar('P', bound='AgglomerativeClustering')
class AgglomerativeClustering:
    def __init__(self) -> None:
        pass

    def _min_distance(self) -> list:
        # 找到不属于一类的两个样本点的最小距离
        # 由于使用的是最短距离来进行类之间的距离计算，所以两不同类点之间的距离只要是最小就可以直接使用
        m = float('inf')
        m_pair = []
        for i in range(self._n):
            for j in range(self._n):
                if i == j or self._c[i] == self._c[j]: continue
                if self._d[i][j] < m:
                    m = self._d[i][j]
                    m_pair = [i ,j]
        return m_pair

    def _distance(self, x1, x2) -> float:
        # 欧式距离
        total = 0.0
        for i in range(self._m):
            total += abs(x1[i] - x2[i]) ** 2
        return total ** 0.5

    def fit(self, X) -> P:
        self._n = len(X)
        self._m = len(X[0])
        self._d = np.zeros((self._n, self._n), dtype=float)
        self._c = np.zeros(self._n, dtype=int)

        # 计算各个样本之间的距离同时得出最近的两个样本进行聚类
        m = float('inf')
        m_pair = []
        for i in range(self._n):
            self._c[i] = i
            for j in range(self._n):
                if i == j or self._d[i][j] > 0: continue
                d = self._distance(X[i], X[j])
                self._d[i][j] = d
                self._d[j][i] = d
                if d < m:
                    m = d
                    m_pair = [i ,j]
        
        self._c[m_pair[1]] = m_pair[0] # ij归位一类
        while len(np.unique(self._c)) > 1 : # 如果未归位一类就继续
            m_pair = self._min_distance()
            j = self._c[m_pair[0]]
            k = self._c[m_pair[1]]
            for i in range(self._n):
                if self._c[i] == j: self._c[i] = k

        return self


from sklearn.datasets import load_iris
iris_dataset = load_iris()
X = iris_dataset['data'][:5]

cluster = AgglomerativeClustering()
cluster.fit(X)
