import math
import numpy as np
from typing import TypeVar


# 适应提升算法
P = TypeVar('P', bound='AdaBoost')
class AdaBoost:
    _classify_functions = []
    _alpha = []
    times = 200
    def __init__(self, times = 200) -> None:
        self.times = 200

    def _error(self, mid, y1, y2) -> float:
        '''
        计算并更新误差最小的分类方法
        mid: 分类间隔
        y1: 正类
        y2: 反类
        '''
        error = 0.0
        for i in range(0, mid): # 属于y1的集合
            if self._y[i] != y1: error += self._omega[i]
        for i in range(mid, self.n): # 属于y2的集合
            if self._y[i] != y2: error += self._omega[i]

        return error

    def _classification(self) -> None:
        self.min_error = 1.0
        classify_function = []
        # 对每个维度都尝试进行分类 寻找误分类最小的点
        for i in range(self.m):
            sorted_index = np.argsort(self.X[:, i]) # 将集合按照某个特征进行排序
            self._X = self.X[sorted_index, :] # 初始化临时集合方便后续计算
            self._y = self.y[sorted_index]
            self._omega = self.omega[sorted_index]

            for j in range(1, self.n):
                error1 = self._error(i, j, 1, -1)
                error2 = self._error(i, j, -1, 1)
                avg = (self._X[j - 1] + self._X[j]) / 2
                if error1 < min_error :
                    min_error = error1
                    classify_function = [i, 1, avg, -1]
                
                if error2 < min_error:
                    min_error = error1
                    classify_function = [i, -1, avg, 1]
        
        return min_error, classify_function

    def fit(self, X, y) -> P:
        if (type(X) is not np.ndarray): X = np.array(X)
        if (type(y) is not np.ndarray): y = np.array(y)

        self.X = X
        self.y = y
        self.n = len(X)
        self.m = len(X[0])
        self.omega = np.array([1 / self.n for _i in range(self.n)]) # 初始化权重分布

        for _i in range(self.times):
            error, classify_function = self._classification()
            self._classify_functions.append(classify_function) # 记录当前最有效的弱分类器
            self._alpha.append(math.log((1 - error) / error) / 2) # 计算当前分类器的权重

            z = 1# 规范化因子



        return self
    
    def predict(self, X_test) -> list:
        res = []
        for x in X_test:
            res.append(np.sign(self.decision_function(x)))
        return np.array(res)

    def score(self, X_test, y_test):
        right = 0
        for i, y in enumerate(self.predict(X_test)):
            if (y == y_test[i]): right += 1
        return right / len(X_test)

from sklearn.datasets import load_iris
iris_dataset = load_iris()
target = []
for y in iris_dataset['target']:
    if y == 0: target.append(-1)
    else: target.append(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'][:100], target[:100], random_state=0
)

svm = AdaBoost()
svm.fit(X_train, y_train)
#print(svm.score(X_test, y_test))