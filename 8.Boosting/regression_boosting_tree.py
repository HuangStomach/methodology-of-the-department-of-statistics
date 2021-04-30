import copy
import numpy as np
from typing import TypeVar

# 适应提升算法 简单实现一维数据的回归树
P = TypeVar('P', bound='RegressionBoostingTree')
class RegressionBoostingTree:
    _regression_trees = [] # 存放各个基本回归树
    def __init__(self, times = 200, epsilon = 0.2) -> None:
        self.times = times
        self.epsilon = epsilon

    def _loss_function(self, real, approximate) -> float:
        # 平方损失函数
        return (real - approximate) ** 2

    def _loss_function_gradient(self, real, approximate) -> float:
        # 计算残差
        return real - approximate

    def _square_error(self, split) -> float:
        # 计算平方误差最小参数c
        left_set = [y for i, y in enumerate(self._y) if self._X[i] < split]
        right_set = [y for i, y in enumerate(self._y) if self._X[i] >= split]
        c1 = np.average(left_set) if len(left_set) > 0 else 0.0
        c2 = np.average(right_set) if len(right_set) > 0 else 0.0
        return c1, c2, sum([self._loss_function(tag, c1) for tag in left_set]) + sum([self._loss_function(tag, c2) for tag in right_set])

    def fit(self, X, y) -> P:
        if (type(X) is not np.ndarray): X = np.array(X)
        if (type(y) is not np.ndarray): y = np.array(y)

        self.X = copy.deepcopy(X)
        self.y = y.copy()
        self.n = len(X)
        self.m = len(X[0])

        # 因为数据的特征都是连续值 计算每一个特征的切分点 每两个数据的中点集合作为这个特征的切点集合
        self.split = []
        for j in range(self.m):
            self.split.append([])  # 将某一个特征的所有切分点放到一个列表中
            _X = self.X[np.argsort(self.X[:, j])]  # 以这个特征的大小来进行排序
            for i in range(self.n - 1):
                sp = (_X[i][j] + _X[i + 1][j]) / 2
                if sp not in self.split[j]:  self.split[j].append(sp)

        for _i in range(self.times):
            # 计算出每个分组的平方误差
            min_error = float("inf")
            regression_trees = []
            for j in range(self.m):
                sorted_index = np.argsort(self.X[:, j]) # 将集合按照某个特征进行排序
                self._X = self.X[sorted_index, j] # 初始化临时集合方便后续计算
                self._y = self.y[sorted_index]
                
                for split in self.split[j]:#遍历每一个切分点
                    c1, c2, error = self._square_error(split)
                    if error >= min_error: continue

                    min_error = error
                    regression_trees = [j, c1, split, c2]

            self._regression_trees.append(regression_trees)
            
            loss = 0.0 # 计算平方损失
            for i in range(self.n):
                self.y[i] = self._loss_function_gradient(y[i], self._predict_once(self.X[i]))
                loss += self.y[i] ** 2
            print(loss)
            if loss <= self.epsilon: break
        
        return self
    
    def _predict_once(self, x) -> float:
        res = 0.0 # 将所有预测值和残差值累加
        for _regression in self._regression_trees:
            f_index, left, split, right = _regression
            if x[f_index] < split: res += left
            else: res += right
        return res
    
    def predict(self, X_test) -> list:
        res = []
        for x in X_test:
            res.append(self._predict_once(x))
        return np.array(res)

    def score(self, X_test, y_test):
        loss = 0.0
        for i, y in enumerate(self.predict(X_test)):
            loss += (y_test[i] - y) ** 2
        return loss

from sklearn.datasets import load_boston
boston_dataset = load_boston()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    boston_dataset['data'][:100], boston_dataset['target'][:100], random_state=5
)

tree = RegressionBoostingTree(2000)
tree.fit(X_train, y_train)
print(tree.predict(X_test))
print(y_test)
