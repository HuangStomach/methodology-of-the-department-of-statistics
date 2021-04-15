# TODO: 实现
import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='CartClassificationTree')
# 目前统计学习方法关于该生成树的构造只有一维特征值的概念
class CartClassificationTree:
    def __init__(self) -> None:
        pass

    def _gini(self, y) -> float:
        # 计算某个集合的基尼指数
        tags = np.unique(y)
        gini = 1.0
        for tag in tags:
            gini -= (y.count(tag) / len(y)) ** 2
        return gini

    def _gini_condition(self, X, y, i) -> dict:
        # 计算在特征i的条件下，集合的基尼指数
        res = {}
        if (type(X) is not np.ndarray):
            X = np.array(X)
        features = np.unique(X[:, i])
        for feature in features:
            # 集合D1 符合该可能取值
            X_eq = [x for x in X if x[i] == feature]
            y_eq = [tag for index, tag in enumerate(y) if X[index][i] == feature]
            # 集合D2 不符合该可能取值
            X_else = [x for x in X if x[i] != feature]
            y_else = [tag for index, tag in enumerate(y) if X[index][i] != feature]
            
            res[feature] = len(X_eq) / len(X) * self._gini(y_eq) + len(X_else) / len(X) * self._gini(y_else)
        return res

    def fit(self, X, y) -> None:
        print(self._gini_condition(X, y, 0))
        pass

X_train = [
    [0, 0, 0, 2],
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 2],
    [0, 0, 0, 2],
    [1, 0, 0, 2],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [2, 0, 1, 0],
    [2, 0, 1, 1],
    [2, 1, 0, 1],
    [2, 1, 0, 0],
    [2, 0, 0, 2],
]

y_train = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

decision = CartClassificationTree()
decision.fit(X_train, y_train)