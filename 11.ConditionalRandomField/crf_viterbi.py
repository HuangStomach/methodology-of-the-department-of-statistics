
import numpy as np
from copy import deepcopy
from typing import TypeVar

# 条件随机场模型学习的维特比算法
P = TypeVar('P', bound='CRFIIS')
class CRFIIS:
    def __init__(self, times = 1, epsilon = 0.001) -> None:
        self._times = times
        self._epsilon = epsilon

    # 特征tk取值方法
    def _feature_t(self, t, y1, y2, i) -> int:
        if y1 == t[0] and y2 == t[1] and i in t[2]: return 1
        return 0

    # 特征sl取值方法
    def _feature_s(self, s, y1, i) -> int:
        if y1 == s[0] and i in s[1]: return 1
        return 0
    
    def _feature_func(self, k, y1, y2, i) -> int:
        if (k < self._k1):
            return self._feature_t(self._t_lambda[k], y1, y2, i) 
        return self._feature_s(self._s_mu[k - self._k1], y2, i)

    

    def fit(self, t_lambda, s_mu) -> P:
        self._t_lambda = t_lambda
        if (type(t_lambda) is not np.ndarray): self._t_lambda = np.array(t_lambda, dtype=object)
        self._s_mu = s_mu
        if (type(s_mu) is not np.ndarray): self._s_mu = np.array(s_mu, dtype=object)

        # 获得标记序列个数m
        self._m = len(np.unique(self._t_lambda[:, 0]))
        self._n = max([max(i) for i in self._t_lambda[:, 2]])
        self._k1 = len(self._t_lambda)
        self._k2 = len(self._s_mu)
        self._omega = np.append(self._t_lambda[:, 3], self._s_mu[:, 2]) # 权值

        self._delta = np.zeros((self._n, self._m))
        self._psi = np.zeros((self._n, self._m), dtype=int)

        # 初始化
        for y in range(self._m):
            F = []
            for k in range(self._k1 + self._k2):
                F.append(self._feature_func(k, 0, y + 1, 1))
            self._delta[0][y] = np.dot(self._omega, F)
        
        # 递推最优路径
        for i in range(1, self._n):
            for l in range(self._m):
                p = []
                for j in range(self._m):
                    F = []
                    for k in range(self._k1 + self._k2):
                        F.append(self._feature_func(k, j + 1, l + 1, i + 1))
                    # 非规范化概率
                    p.append(np.dot(self._omega, F))
                self._delta[i][l] = np.max(p)
                self._psi[i][l] = np.argmax(p)
        
        return self

    def optimal_path(self) -> list:
        path = []
        index = np.argmax(self._delta[self._n - 1])
        path.append(index + 1)
        
        for i in range(self._n - 1, 0, -1):
            index = self._psi[i][index]
            path.append(index + 1)
        path.reverse()

        return path

crf = CRFIIS()
# t_lambda [[y1, y2, i=(...)], lambda]
t_lambda = [
    [1, 2, [2, 3], 1],
    [1, 1, [2], 0.6],
    [2, 1, [3], 1],
    [2, 1, [2], 1],
    [2, 2, [3], 0.2]
]

s_mu = [
    [1, [1], 1],
    [2, [1, 2], 0.5],
    [1, [2, 3], 0.8],
    [2, [3], 0.5]
]

crf.fit(t_lambda, s_mu)
print(crf.optimal_path())
