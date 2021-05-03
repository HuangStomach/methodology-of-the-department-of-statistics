import math
import numpy as np
from typing import TypeVar

# 适应提升算法 简单实现一维数据的回归树
P = TypeVar('P', bound='HMMBaumWelch')
# 向前算法
class HMMBaumWelch:
    def __init__(self, times, epsilon) -> None:
        self._times = times
        self._epsilon = epsilon
    
    def init(self, A, B, pi) -> P:
        self._A = A
        self._B = B
        self._pi = pi
        return self

    def _em(self) -> P:
        # E步 依据当前模型参数 计算分模型k对观测数据yi的响应度
        _gamma = np.zeros((self._t, self.n), dtype = float)
        _xi = np.zeros(self._t)

        # E步 计算时刻t处于状态i的概率
        for t in range(self._t):
            total = 0.0
            for i in range(self._n):
                total += self._alpha[t][i] * self._beta[t][i]
            for i in range(self._n):
                _gamma[t][i] = self._alpha[t][i] * self._beta[t][i] / total
        
        # E步 计算时刻t处于状态i且时刻t+1处于状态j的概率
        for t in range(self._t - 1):
            _xi[t] = np.zeros((self._n, self._n))
            total = 0.0
            for i in range(self._n):
                for j in range(self._n):
                    total += self._alpha[t][i] * self._A[i][j] * self._B[O[t + 1]] * self._beta[t + 1][j]

            for i in range(self._n):
                for j in range(self._n):
                    _xi[t][i][j] = self._alpha[t][i] * self._A[i][j] * self._B[O[t + 1]] * self._beta[t + 1][j] / total

        # M 步计算新一轮迭代的模型参数
        self._alpha_old = self._alpha.copy() # 不同高斯模型下的系数
        self._mu_old = self._mu.copy() # 期望初值
        self._sigma_old = self._sigma.copy() # 方差初值

        for j in range(self._k):
            self._mu[j] = sum([g * self.y[i] for i, g in enumerate(_gamma[j])]) / sum(_gamma[j])
            self._sigma[j] = sum([g * (self.y[i] - self._mu[j]) ** 2 for i, g in enumerate(_gamma[j])]) / sum(_gamma[j])
            self._alpha[j] = sum(_gamma[j]) / self.n
        pass

    def fit(self, O) -> P:
        '''
        O: Observation sequence 观测序列
        '''
        self._O = O
        self._t = len(O)
        self._n = len(self._A)
        self._alpha = np.zeros((self._t, self._n), dtype = float)
        self._beta = np.ones((self._t, self._n), dtype = float)

        # 初始化前向概率
        for j in range(self._n):
            self._alpha[0][j] = self._pi[j] * self._B[j][O[0]]
        
        for i in range(1, self._t):
            for j in range(self._n):
                self._alpha[i][j] = self._B[j][O[i]]
                self._alpha[i][j] *= sum([alpha * self._A[i][j] for i, alpha in enumerate(self._alpha[i - 1])])

        # 初始化后向概率
        for i in range(self._t - 2, -1, -1):
            for j in range(self._n):
                self._beta[i][j] = 0.0
                for k in range(self._n):
                    self._beta[i][j] += self._A[j][k] * self._B[k][O[i + 1]] * self._beta[i + 1][j]
        
        for _i in range(self._times):
            self._em()
            if self._convergence(): break
        
        return self

A = [
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
]
B = [
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
]
pi = [0.2, 0.4, 0.4]
O = [0, 1, 0, 0, 1, 0, 1, 1]

hmm = HMMBaumWelch()
hmm.init(A, B, pi).fit(O)
# print(hmm.sequence())