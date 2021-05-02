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

    def fit(self, O, n) -> P:
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