from copy import deepcopy
import math
import numpy as np

# PageRank的迭代算法
class PageRank:
    def __init__(self, d, epsilon = 0.0001) -> None:
        '''
        PageRank 迭代算法
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
            if A[i] - B[i] >= self._epsilon: return False
        return True
    
    def transform(self, M) -> None:
        '''
        PageRank 向量计算函数
        ### Parameters
        * M 转移矩阵
        '''
        self._M = self._d * self._transpose(M)

        # 初始化平滑项  
        self._n = len(self._M)
        smooth = np.zeros(self._n)
        R = np.zeros(self._n)
        old_R = np.zeros(self._n)
        for i in range(self._n): 
            smooth[i] = (1 - self._d) / self._n
            R[i] = 1 / self._n
        
        while self._convergence(R, old_R) != True:
            old_R = R.copy()
            R = np.dot(self._M, R) + smooth

        return R

M = np.array([
    [0, 1/3, 1/3, 1/3],
    [1/2, 0, 0, 1/2],
    [0, 0, 1, 0],
    [0, 1/2, 1/2, 0],
])
pagerank = PageRank(0.8)
R = pagerank.transform(M)
print(R)
