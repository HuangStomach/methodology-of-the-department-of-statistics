import math
import numpy as np
from copy import deepcopy

# 改进的迭代尺度算法IIS
# 参考 https://blog.csdn.net/breeze_blows/article/details/86612834
class MaximumEntropy:
    _exp_eps = {}   # key为(x,y)，value为出现次数
    _N = 0          # 样本数
    _M = 0          # 特征总数
    _w = {}         # 权重向量 对应每一对特征和标签
    def __init__(self, EPS = 0.005):
        self._EPS = EPS     # 收敛条件

    def _Zx(self, X):
        # 计算每个特征值的Z(x)值
        zx = 0
        for y in self._y:
            ss = 0
            for x in X:
                if (x, y) in self._exp_eps: 
                    ss += self._w[(x, y)]
            zx += math.exp(ss)
        return zx

    def _conditional_probability(self, y, X):
        # 计算每个P(y|x) 既该条件下识别为y的概率
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._exp_eps: ss += self._w[(x, y)]
        res = math.exp(ss) / zx
        return res
 
    def _condition_ep(self, x, y):
        # 计算特征函数fi关于模型的期望
        ep = 0
        for sample in self._X:
            if x not in sample: continue
            conditional_probability = self._conditional_probability(y, sample)
            ep += conditional_probability / self._N
        return ep

    def _convergence(self, lastw):  # 判断是否全部收敛
        for last, now in zip(lastw.values(), self._w.values()):
            if abs(last - now) >= self._EPS: return False
        return True

    def fit(self, X, y, maxiter = 1000) -> None:
        self._X = deepcopy(X)
        self._y = np.unique(y)
        self._N = len(self._X)

        for i, x in enumerate(self._X):
            for feature in x:
                self._exp_eps.setdefault((feature, y[i]), 0)
                self._exp_eps[(feature, y[i])] += 1

        M = max([len(x) for x in self._X])
        for xy, times in self._exp_eps.items():   # 计算特征函数fi关于经验分布的期望
            self._exp_eps[xy] = times / self._N
            self._w[xy] = 0

        for i in range(maxiter):  # 最大训练次数
            lastw = self._w.copy()
            for (x, y), exp_ep in self._exp_eps.items():
                condition_ep = self._condition_ep(x, y) # 计算第i个特征的模型期望
                self._w[(x, y)] += math.log(exp_ep / condition_ep) / M # 更新参数

            if self._convergence(lastw): break # 判断是否收敛

    def predict(self, X): # 计算预测概率
        res = []
        for x in X:
            result = {}
            for y in self._y:
                result[y] = self._conditional_probability(y, x)
            res.append(result)

        return res

X_train = [
    ['sunny', 'hot', 'high', 'FALSE'],
    ['sunny', 'hot', 'high', 'TRUE'],
    ['overcast', 'hot', 'high', 'FALSE'],
    ['rainy', 'mild', 'high', 'FALSE'],
    ['rainy', 'cool', 'normal', 'FALSE'],
    ['rainy', 'cool', 'normal', 'TRUE'],
    ['overcast', 'cool', 'normal', 'TRUE'],
    ['sunny', 'mild', 'high', 'FALSE'],
    ['sunny', 'cool', 'normal', 'FALSE'],
    ['rainy', 'mild', 'normal', 'FALSE'],
    ['sunny', 'mild', 'normal', 'TRUE'],
    ['overcast', 'mild', 'high', 'TRUE'],
    ['overcast', 'hot', 'normal', 'FALSE'],
    ['rainy', 'mild', 'high', 'TRUE']
]
y_train = [-1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1]

maxent = MaximumEntropy()
x_test = [
    ['overcast', 'mild', 'high', 'FALSE']
]
maxent.fit(X_train, y_train)
print('predict:', maxent.predict(x_test))
