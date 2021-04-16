import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='KNearestNeighbor')
class KNearestNeighbor:
    k = 5
    p = 2
    def __init__(self, k = 5, p = 2) -> None:
        '''
        k: 选取几个近邻进行判断
        p: 两者距离计算的度量
        '''
        self.k = k
        self.p = p

    def _minkowski_distance(self, X1, X2) -> float:
        # 根据p值计算Minkowski距离
        sum = 0.0
        for i, j in zip(X1, X2):
            sum += abs(i - j) ** self.p
            
        return sum ** (1 / self.p)

    def _chebyshev_distance(self, X1, X2) -> float:
        # 如果p值为-1 则计算切比雪夫距离
        dis = []
        for i, j in zip(X1, X2):
            dis.append(abs(i - j))
            
        return max(dis)

    def fit(self, X, y) -> P:
        # 基础knn算法不进行训练，只对训练数据进行存储
        self.X = X
        self.y = y
        return self

    def _predict_one(self, item) -> int:
        # 计算距离并取最近K个距离进行分类识别
        dis = []
        for i, x in enumerate(self.X):
            distance = self._minkowski_distance(x, item) if self.p > 0 else self._chebyshev_distance(x, item)
            dis.append((distance, self.y[i]))
        dis = sorted(dis, key = lambda x : x[0])

        # 取出前k个近邻并求类别中的众数
        res = []
        for i in range(self.k):
            res.append(dis[i][1])

        counts = np.bincount(res)
        return np.argmax(counts)
    
    def predict(self, Test) -> list:
        res = []
        for item in Test:
            res.append(self._predict_one(item))
        return np.array(res)

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

knn = KNearestNeighbor(7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)