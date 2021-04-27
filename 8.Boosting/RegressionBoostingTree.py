import math
import numpy as np
from typing import TypeVar

# 适应提升算法 简单实现一维数据的回归树
P = TypeVar('P', bound='RegressionBoostingTree')
class RegressionBoostingTree:
    _regression_trees = [] # 存放各个基本回归树
    times = 200
    def __init__(self, times = 200) -> None:
        self.times = times

    def _loss_function(self, real, approximate) -> float:
        # 平方损失函数
        return (real - approximate) ** 2

    def _loss_function_gradient(self, real, approximate) -> float:
        # 平方损失函数负梯度
        return (real - approximate)

    def _square_error(self, mid) -> float:
        # 计算平方误差最小参数c
        left_set = self._y[0:mid + 1]
        right_set = self._y[mid:self.n]
        c1 = np.average(left_set) if len(left_set) > 0 else 0.0
        c2 = np.average(right_set) if len(right_set) > 0 else 0.0
        return c1, c2, sum([self._loss_function(tag, c1) for tag in left_set]) + sum([self._loss_function(tag, c2) for tag in right_set])

    def fit(self, X, y) -> P:
        if (type(X) is not np.ndarray): X = np.array(X)
        if (type(y) is not np.ndarray): y = np.array(y)

        self.X = X
        self.y = y
        self.n = len(X)
        self.m = len(X[0])

        for _i in range(6):
            # 计算出每个分组的平方误差
            min_error = float("inf")
            regression_tree = []
            for j in range(self.m):
                for i in range(self.n - 1):
                    sorted_index = np.argsort(self.X[:, j]) # 将集合按照某个特征进行排序
                    self._X = self.X[sorted_index, j] # 初始化临时集合方便后续计算
                    self._y = self.y[sorted_index]

                    c1, c2, error = self._square_error(i)
                    if error >= min_error: continue

                    min_error = error
                    split = (self._X[i] + self._X[i + 1]) / 2
                    regression_tree = [
                        [j, c1, float('-inf'), split],
                        [j, c2, split, float('inf')]
                    ]

            self._regression_trees += regression_tree
            
            for i in range(self.n):
                self.y[i] = self._loss_function_gradient(self.y[i], self._predict_once(self.X[i]))
            print(self.y)
        # c = self._regression_trees[0][1] # 使损失函数极小化的值c

        '''
        for _i in range(3):
            r = []
            for i in range(self.n):
                r.append(self._loss_function_gradient(self.y[i], self._predict_once(self.X[i])))
            print(r)
        '''
        print(self._regression_trees)

        return self
    
    def _predict_once(self, x) -> float:
        for _regression in self._regression_trees:
            f_index, value, left, right = _regression
            if x[f_index] >= left and x[f_index] < right:
                return value
        return 0.0
    
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
    boston_dataset['data'][:, 1], boston_dataset['target']
)

X_train = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y_train = [5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9.0, 9.05]
tree = RegressionBoostingTree()
tree.fit(X_train, y_train)
#print(svm.score(X_test, y_test))
