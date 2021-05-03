
import numpy as np
from copy import deepcopy
from typing import TypeVar

# 适应提升算法 简单实现一维数据的回归树
P = TypeVar('P', bound='HMMBaumWelch')
# 向前算法
class HMMBaumWelch:
    def __init__(self, times = 1, epsilon = 0.001) -> None:
        self._times = times
        self._epsilon = epsilon
    
    def init(self, A, B, pi) -> P:
        self._A = A
        self._B = B
        self._pi = pi
        return self

    def _em(self) -> P:
        # E步 依据当前模型参数 计算分模型k对观测数据yi的响应度
        _gamma = np.zeros((self._t, self._n), dtype = float)
        _xi = []

        # E步 计算时刻t处于状态i的概率
        for t in range(self._t):
            total = 0.0
            for i in range(self._n):
                total += self._alpha[t][i] * self._beta[t][i]
            for i in range(self._n):
                _gamma[t][i] = self._alpha[t][i] * self._beta[t][i] / total
        
        # E步 计算时刻t处于状态i且时刻t+1处于状态j的概率
        for t in range(self._t - 1):
            xi = np.zeros((self._n, self._n))
            total = 0.0
            for i in range(self._n):
                for j in range(self._n):
                    total += self._alpha[t][i] * self._A[i][j] * self._B[j][O[t + 1]] * self._beta[t + 1][j]

            for i in range(self._n):
                for j in range(self._n):
                    xi[i][j] = self._alpha[t][i] * self._A[i][j] * self._B[j][O[t + 1]] * self._beta[t + 1][j] / total
            _xi.append(xi)

        # M 步计算新一轮迭代的模型参数
        for i in range(self._n):
            b = 0.0
            for t in range(self._t - 1):
                b += _gamma[t][i]

            for j in range(self._n):
                a = 0.0
                for t in range(self._t - 1):
                    a += _xi[t][i][j]
                self._A[i][j] = a / b
        
        for i in range(self._n):
            b = 0.0
            for t in range(self._t):
                b += _gamma[t][i]

            for k in range(self._k):
                a = 0.0
                for t in range(self._t):
                    a += _gamma[t][i] * (1 if self._O[t] == k else 0)
                self._B[i][k] = a / b

            self._pi[i] = _gamma[0][i]

    def _convergence(self) -> bool:
        for i in range(self._n):
            for j in range(self._n):
                if abs(self._A[i][j] - self._A_old[i][j]) > self._epsilon: return False
            
            for k in range(self._k):
                if abs(self._B[i][k] - self._B_old[i][k]) > self._epsilon: return False
            if abs(self._pi[i] - self._pi_old[i]) > self._epsilon: return False
        
        return True

    def fit(self, O) -> P:
        '''
        O: Observation sequence 观测序列
        '''
        self._O = O
        self._t = len(O) # 时间序列值
        self._n = len(self._A) # 状态个数
        self._k = len(self._B[0]) # 观测值种类个数
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
        
        self._A_old = deepcopy(self._A)
        self._B_old = deepcopy(self._B)
        self._pi_old = self._pi.copy()
        
        for _i in range(self._times):
            self._em()
            if self._convergence(): break
        
        return self

A = [
    [0.1, 0.7, 0.2],
    [0.9, 0.05, 0.05],
    [0.5, 0.2, 0.3]
]
B = [
    [0.4, 0.6],
    [0.3, 0.7],
    [0.3, 0.7]
]
pi = [0.4, 0.1, 0.5]
O = [0, 1, 0, 0, 1, 0, 1, 1]

hmm = HMMBaumWelch()
hmm.init(A, B, pi).fit(O)
print(hmm._A)
print(hmm._B)
print(hmm._pi)