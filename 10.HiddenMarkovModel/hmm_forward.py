import numpy as np
from typing import TypeVar

# 适应提升算法 简单实现一维数据的回归树
P = TypeVar('P', bound='HMMForward')
# 向前算法
class HMMForward:
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

        # 初始化初值
        for j in range(self._n):
            self._alpha[0][j] = self._pi[j] * self._B[j][O[0]]
        
        for i in range(1, self._t):
            for j in range(self._n):
                self._alpha[i][j] = self._B[j][O[i]]
                self._alpha[i][j] *= sum([alpha * self._A[i][j] for i, alpha in enumerate(self._alpha[i - 1])])
        
        return self

    def sequence(self) -> float:
        return sum(self._alpha[self._t - 1])

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
hmm = HMMForward(A, B, pi)
hmm.fit([0, 1, 0])
print(hmm.sequence())