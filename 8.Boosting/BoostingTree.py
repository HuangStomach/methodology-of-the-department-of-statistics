import math
import numpy as np
from typing import TypeVar

# 适应提升算法
P = TypeVar('P', bound='BoostingTree')
class BoostingTree:
    times = 200
    def __init__(self, times = 200) -> None:
        self.times = times

    def fit(self, X, y) -> P:
        if (type(X) is not np.ndarray): X = np.array(X)
        if (type(y) is not np.ndarray): y = np.array(y)

        self.X = X
        self.y = y
        self.n = len(X)
        self.m = len(X[0])

        return self
    
    def predict(self, X_test) -> list:
        res = []
        for x in X_test:
            f = 0.0
            for i in range(len(self._classify_functions)):
                f += self._alpha[i] * self._classify(x, i)
            res.append(np.sign(f))
        return np.array(res)

    def score(self, X_test, y_test):
        right = 0
        for i, y in enumerate(self.predict(X_test)):
            if (y == y_test[i]): right += 1
        return right / len(X_test)

from sklearn.datasets import load_boston
boston_dataset = load_boston()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    boston_dataset['data'], boston_dataset['target']
)

svm = BoostingTree()
svm.fit(X_train, y_train)
#print(svm.score(X_test, y_test))
