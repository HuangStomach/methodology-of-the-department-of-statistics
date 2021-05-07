
import numpy as np
from copy import deepcopy
from typing import TypeVar

# 适应提升算法 简单实现一维数据的回归树
P = TypeVar('P', bound='HMMViterbi')
# 向前算法
class HMMViterbi:
    def __init__(self, A, B, pi) -> None:
        self._A = A
        self._B = B
        self._pi = pi
    
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

        # 初始化第一步观测的概率
        self._delta = np.zeros((self._t, self._n))
        self._psi = np.zeros((self._t, self._n), dtype=int)
        for i in range(self._n):
            self._delta[0][i] = self._pi[i] * self._B[i][O[0]]

        # 遍历计算每个时间点每个状态的最大概率
        for t in range(1, self._t):
            for i in range(self._n):
                temp = [self._delta[t - 1][j] * self._A[j][i] for j in range(self._n)]
                self._delta[t][i] = max(temp) * self._B[i][O[t]]
                self._psi[t][i] = temp.index(max(temp))
        
        return self

    def optimal_path(self) -> dict:
        p = max(self._delta[self._t - 1])
        index = np.argmax(self._delta[self._t - 1])
        path = []
        path.append(index)
        
        for t in range(self._t - 1, 0, -1):
            index = self._psi[t][index]
            path.append(index)
        path.reverse()

        return {
            'probability': p,
            'path': path
        }
    

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
O = [0, 1, 0]

hmm = HMMViterbi(A, B, pi).fit(O)
hmm.fit(O)
print(hmm.optimal_path())