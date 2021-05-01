import math
import numpy as np

# 向前算法
class HMMForward:
    def __init__(self, A, B, pi) -> None:
        self._A = A
        self._B = B
        self._pi = pi

    def calculate(self, O) -> float:
        '''
        O: Observation sequence 观测序列
        '''
        t = len(O)
        n = len(self._A)
        self._alpha = np.zeros((t, n), dtype = float)

        # 初始化初值
        for j in range(n):
            self._alpha[0][j] = self._pi[j] * self._B[j][O[0]]
        
        for i in range(1, t):
            for j in range(n):
                self._alpha[i][j] = self._B[j][O[i]]
                self._alpha[i][j] *= sum([alpha * self._A[i][j] for i, alpha in enumerate(self._alpha[i - 1])])
        
        return sum(self._alpha[t - 1])

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
print(hmm.calculate([0, 1, 0]))