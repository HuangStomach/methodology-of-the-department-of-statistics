from copy import deepcopy
import math
import numpy as np

# PageRank的幂法
class PageRank:
    def __init__(self, d, epsilon = 0.0001) -> None:
        '''
        PageRank 幂法
        ### Parameters
        * d 阻尼因子
        * epsilon 收殓判别参数
        '''
        self._d = d
        self._epsilon = epsilon
    
    def _transpose(self, X) -> list:
        '''
        转置矩阵
        ### Parameters
        * X 需转置矩阵
        '''
        n = len(X)
        m = len(X[0])
        _X = np.zeros((m, n), dtype=float)
        for i in range(n):
            for j in range(m):
                _X[j][i] = X[i][j]
        return _X

    def _convergence(self, A, B) -> bool:
        '''
        判别是否收殓
        * A 判敛矩阵1
        * B 判敛矩阵2
        '''
        for i in range(self._n):
            if abs(A[i] - B[i]) >= self._epsilon: return False
        return True
    
    def transform(self, M) -> None:
        '''
        PageRank 向量计算函数
        ### Parameters
        * M 转移矩阵
        '''
        self._M = self._transpose(M)

        # 初始转移向量
        self._n = len(self._M)
        old_X = np.ones(self._n)
        X = np.zeros(self._n)

        # 一般转移矩阵
        A = self._d * self._M + (1 - self._d) / self._n * np.ones((self._n, self._n))
        
        while True:
            y = np.dot(A, old_X)
            norm = max([abs(x) for x in y])
            X = y / norm
            if self._convergence(X, old_X): break
            old_X = X.copy()
        
        return X

M = np.array([
    [0, 1/2, 1/2],
    [0, 0, 1],
    [1, 0, 0]
])
pagerank = PageRank(0.85)
R = pagerank.transform(M)
print(R)
