import numpy as np
np.set_printoptions(suppress=True)
from copy import deepcopy
from typing import TypeVar

# 主成分分析奇异值分解算法
P = TypeVar('P', bound='PCA')
class PCA:
    def __init__(self, threshold) -> None:
        self._threshold = threshold

    def standardize(self, X) -> list: # 标准化 使样本元素处于均值0方差1范围内
        n = len(X)
        m = len(X[0])

        _X = []
        for i in range(m):
            mean = np.mean(X[:,i])
            sii = np.sum((X[:,i] - mean) ** 2) / (n - 1)
            if sii > 0: _X.append((X[:, i] - mean) / sii)
            else: _X.append(X[:, i] - mean)
        return self._transpose(_X)
    
    def _transpose(self, X) -> list: # 转置矩阵
        n = len(X)
        m = len(X[0])
        _X = np.zeros((m, n), dtype=float)
        for i in range(n):
            for j in range(m):
                _X[j][i] = X[i][j]
        return _X

    def transform(self, X) -> list:
        m = len(X[0])
        _X = X / (m - 1) # 数据的表示方式实际上就为样本集合的转置
        _Xt = self._transpose(_X)
        Sx = np.dot(_Xt, _X)
        _lambda, _alpha = np.linalg.eig(Sx)

        sorted_index = np.argsort(_lambda)[::-1] # 将集合按照特征进行排序
        _lambda = _lambda[sorted_index]
        _Vt = _alpha[sorted_index] # V的列向量就是协方差矩阵的单位特征向量

        total_lambda = sum(_lambda)
        k = 0
        add = 0.0
        for i in _lambda: # 得出符合条件的k
            add += i
            k += 1
            if add / total_lambda >= self._threshold: break

        return np.dot(_Vt[:k], self._transpose(X)) # 返回主成分矩阵

X_train = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 4],
    [0, 3, 0, 0],
    [0, 0, 0, 0],
    [2, 0, 0, 0]
])
pca = PCA(0.8)
Y = pca.transform(pca.standardize(X_train))
print(Y)
