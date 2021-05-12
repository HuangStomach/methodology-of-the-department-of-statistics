from copy import deepcopy
import math
import numpy as np

# 概率潜在语义模型参数估计的EM算法
class PLSA:
    _times = 200
    def __init__(self, k, times = 200, epsilon = 0.5) -> None:
        self._k = k # 话题个数
        self._times = times # 最大迭代次数
        self._epsilon = epsilon # 接受误差范围
    
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

    def _convergence(self, w_z, z_d) -> bool:
        for k in range(self._k):
            for j in range(self._m):
                if abs(w_z[k][j] - self._old_wz[k][j]) > self._epsilon: return False
            for i in range(self._n):
                if abs(z_d[i][k] - self._old_zd[i][k]) > self._epsilon: return False
        
        return True

    def fit_predict(self, X) -> dict:
        self._m = len(X) # 单词集合长度
        self._n = len(X[0]) # 文本集合长度

        z_wd = np.zeros((self._k, self._n, self._m)) # P(z|w,d)
        w_z = np.random.rand(self._k, self._m) # P(w|z)
        z_d = np.random.rand(self._n, self._k) # P(z|d)

        for _i in range(self._times):
            self._old_wz = deepcopy(w_z)
            self._old_zd = deepcopy(z_d)
            # E步
            for i in range(self._n):
                for j in range(self._m):
                    total = 0.0
                    for k in range(self._k): total += z_d[i][k] * w_z[k][j]
                    for k in range(self._k):
                        z_wd[k][i][j] =  z_d[i][k] * w_z[k][j]
                        if total != 0: z_wd[k][i][j] /= total

            # M步       
            for k in range(self._k):
                total = 0.0
                for j in range(self._m):
                    for i in range(self._n): total += X[j][i] * z_wd[k][i][j]
                
                for j in range(self._m):
                    w_z[k][j] = 0.0
                    for i in range(self._n): 
                        w_z[k][j] += X[j][i] * z_wd[k][i][j]
                    if total != 0: w_z[k][j] /= total
            
            for i in range(self._n):
                n = sum(X[:, i])
                for k in range(self._k):
                    total = 0.0
                    for j in range(self._m): total += X[j][i] * z_wd[k][i][j]
                    w_z[k][j] = total
                    if n != 0: w_z[k][j] /= n

            if self._convergence(w_z, z_d): break
        
        return {
            'w_z': w_z,
            'z_d': z_d
        }

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
plsa = PLSA(3)
print(plsa.fit_predict(X))
