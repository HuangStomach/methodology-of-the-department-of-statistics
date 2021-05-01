import math
import numpy as np
from typing import TypeVar

# 适应提升算法 简单实现一维数据的回归树
P = TypeVar('P', bound='HMMForwardBackward')

# 向前算法
class HMMForwardBackward:
    def __init__(self, A, B, pi) -> None:
        self._A = A
        self._B = B
        self._pi = pi

    def fit(self, O) -> P:
        '''
        O: Observation sequence 观测序列
        '''
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
        
        return self

    def status(self, t, i) -> float:
        '''
        在时间t处于状态i的概率
        '''
        t_i = self._alpha[t - 1][i - 1]
        total = 0.0
        for i in range(self._n):
            total += self._alpha[t - 1][i]
        return t_i / total

A = [
    [0.5, 0.1, 0.4],
    [0.3, 0.5, 0.2],
    [0.2, 0.2, 0.6]
]

B = [
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
]

pi = [0.2, 0.3, 0.5]
hmm = HMMForwardBackward(A, B, pi)
hmm.fit([0, 1, 0, 0, 1, 0, 1, 1])
print(hmm.status(4, 3))
