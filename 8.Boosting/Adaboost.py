import math
import numpy as np
from typing import TypeVar


# 适应提升算法
P = TypeVar('P', bound='AdaBoost')
class AdaBoost:
    _classify_functions = []
    _alpha = []
    _result = [] # 记录误差率和分类函数结果之积 
    times = 200
    def __init__(self, times = 200) -> None:
        self.times = times

    def _error(self, mid, y1, y2) -> float:
        '''
        计算并更新误差最小的分类方法
        mid: 分类间隔
        y1: 正类
        y2: 反类
        '''
        error = 0.0
        for i in range(self.n):
            y = y1 if i < mid else y2
            if self._y[i] != y: error += self._omega[i]
        
        return error

    def _classification(self) -> None:
        min_error = 1.0
        classify_function = []
        # 对每个维度都尝试进行分类 寻找误分类最小的点
        for i in range(self.m):
            sorted_index = np.argsort(self.X[:, i]) # 将集合按照某个特征进行排序
            self._X = self.X[sorted_index, i] # 初始化临时集合方便后续计算
            self._y = self.y[sorted_index]
            self._omega = self.omega[sorted_index]
            for j in range(1, self.n):
                error1 = self._error(j, 1, -1)
                error2 = self._error(j, -1, 1)
                avg = (self._X[j - 1] + self._X[j]) / 2
                
                if error1 < min_error :
                    min_error = error1
                    classify_function = [i, 1, avg, -1]
                
                if error2 < min_error:
                    min_error = error2
                    classify_function = [i, -1, avg, 1]
        
        return min_error, classify_function

    # 使用相应的基本分类函数进行分类
    def _classify(self, x, i) -> int:
        feature_index, tag1, threshold, tag2 = self._classify_functions[i]
        feature = x[feature_index] # 取得需要判别的特征值
        if feature <= threshold: return tag1
        else: return tag2

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
            index = len(self._alpha) # 当前轮次
            self._classify_functions.append(classify_function) # 记录当前最有效的弱分类器

            if error == 0.0: self._alpha.append(1.0) # 该分类完整无缺的进行分类 则默认权重最大
            else: self._alpha.append(math.log((1 - error) / error) / 2) # 计算当前分类器的权重

            self._result.append([])

            for i, x in enumerate(self.X):
                self._result[index].append(self._alpha[index] * self._classify(x, index))
    
            misclassification = 0 # 记录误分类点个数
            for i in range(self.n):
                f = 0.0
                for j in range(len(self._classify_functions)):
                    f += self._result[j][i]
                if np.sign(f) != y[i]: misclassification += 1

            if misclassification == 0: break # 无误分类点，训练结束

            # 否则继续计算
            weight = []
            for i, x in enumerate(self.X):
                weight.append(self.omega[i] * math.exp(-1 * y[i] * self._result[index][i]))
            z = sum(weight) # 规范化因子

            # 更新权重
            for i in range(self.n):
                self.omega[i] = weight[i] / z 

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


from sklearn.datasets import load_iris
iris_dataset = load_iris()
target = []
for y in iris_dataset['target']:
    if y == 0: target.append(-1)
    else: target.append(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'][:100], target[:100]
)

svm = AdaBoost()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))
