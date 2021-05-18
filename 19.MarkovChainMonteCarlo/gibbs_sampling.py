from copy import deepcopy
import math
import numpy as np

# 吉布斯 二元正太分布抽样
class MCMC:
    _times = 200
    def __init__(self, m = 1980, n = 2000) -> None:
        '''
        吉布斯 二元正太分布抽样
        ### Parameters
        * m 收殓步数
        * n 迭代步数
        '''
        self._m = m
        self._n = n

    def _random_sample(self) -> float:
        pass 

    def sample(self, mean, cov) -> list:
        self._mean = [mean, mean]
        self._cov = [[1, cov], [cov , 1]]

        res = []
        # 初始化X0
        x = [np.random.multivariate_normal(self._mean, self._cov), np.random.multivariate_normal(self._mean, self._cov)]
        for _i in range(self._n):
            old = x.copy()
            x[0] = np.random.normal(cov * old[1], 1 - cov * cov)
            x[1] = np.random.normal(cov * old[0], 1 - cov * cov)
            if _i >= self._m: res.append(x.copy())

        return res

mcmc = MCMC()
print(mcmc.sample(0, 0.5))
