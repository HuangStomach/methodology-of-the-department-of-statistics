import numpy as np
np.set_printoptions(suppress=True)
from copy import deepcopy
from typing import TypeVar

# 非负矩阵分解算法
P = TypeVar('P', bound='NMF')
class NMF:
    def __init__(self, k, times = 200) -> None:
        '''
        非负矩阵分解算法
        ### Parameters
        * k 文本集合的话题个数
        * times 最大迭代次数
        '''
        self._k = k
        self._times = times
    
    def _normalize(self, X) -> list:
        '''
        归一化 使列向量为单位向量
        ### Parameters
        * X 需归一化矩阵
        '''
        m = len(X[0])

        _X = []
        for i in range(m):
            total = sum(x ** 2 for x in X[:,i]) ** 0.5
            _X.append(X[:, i] / total)
        return self._transpose(_X)
    
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

    def transform(self, Xt) -> list:
        '''
        矩阵分解
        '''
        self._n = len(Xt)
        self._m = len(Xt[0])
        X = self._transpose(Xt)  # 按照惯用数据格式，传递进来的矩阵实际上是待处理矩阵的转置
        W = 10 * np.random.random_sample((self._m, self._k))
        H = 10 * np.random.random_sample((self._k, self._n))

        for _i in range(self._times):
            W = self._normalize(W)
            W_old = deepcopy(W)
            Wt = self._transpose(W)
            Ht = self._transpose(H)

            XHt = np.dot(X, Ht)
            WHHt = np.dot(np.dot(W, H), Ht)
            WtX = np.dot(Wt, X)
            WtWH = np.dot(np.dot(Wt, W), H)

            for l in range(self._k):
                for i in range(self._m):
                    W[i][l] = W[i][l] * XHt[i][l] / WHHt[i][l]
                for j in range(self._n):
                    H[l][j] = H[l][j] * WtX[l][j] / WtWH[l][j]
            
            if (W == W_old).all(): break

        return W, H

# ['book', 'dads', 'dummies', 'estate', 'guide', 'inveesting', 'market', 'real', 'rich', 'stock', 'value']
X = np.array([
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
])
nmf = NMF(3)
W, H = nmf.transform(X)
print(W)
