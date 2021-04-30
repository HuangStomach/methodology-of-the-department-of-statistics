import math
import numpy as np

# 适应提升算法
class EMGaussain:
    _times = 200
    def __init__(self, k, alpha, mu, sigma, times = 200, epsilon = 0.5) -> None:
        self._k = k # 分量个数
        self._alpha = [alpha] * self._k
        self._mu = mu
        self._sigma = [sigma] * self._k
        self._times = times # 最大迭代次数
        self._epsilon = epsilon # 接受误差范围

    def _distribution_density(self, y, k) -> float:
        # 相应分量的高斯分布密度
        return 1 / (2 * math.pi * self._sigma[k]) ** 0.5 * math.exp(
            -1 * (y - self._mu[k]) ** 2 / (2 * self._sigma[k])
        )

    def _convergence(self) -> bool:
        for j in range(self._k):
            if abs(self._mu[j] - self._mu_old[j]) > self._epsilon: return False
            if abs(self._sigma[j] - self._sigma_old[j]) > self._epsilon: return False
            if abs(self._alpha[j] - self._alpha_old[j]) > self._epsilon: return False
        
        return True

    def fit_predict(self, y) -> dict:
        if (type(y) is not np.ndarray): X = np.array(y)

        self.y = y.copy()
        self.n = len(X)

        for _i in range(self._times):
            # E步 依据当前模型参数 计算分模型k对观测数据yi的响应度
            _gamma = np.zeros((self._k, self.n), dtype = float)
            for i in range(self.n):
                _sum = 0.0
                for j in range(self._k):
                    _gamma[j][i] = self._alpha[j] * self._distribution_density(self.y[i], j)
                    _sum += _gamma[j][i]
                for j in range(self._k):
                    if _sum > 0.0: _gamma[j][i] = _gamma[j][i] / _sum

            # M 步计算新一轮迭代的模型参数
            self._alpha_old = self._alpha.copy() # 不同高斯模型下的系数
            self._mu_old = self._mu.copy() # 期望初值
            self._sigma_old = self._sigma.copy() # 方差初值

            for j in range(self._k):
                self._mu[j] = sum([g * self.y[i] for i, g in enumerate(_gamma[j])]) / sum(_gamma[j])
                self._sigma[j] = sum([g * (self.y[i] - self._mu[j]) ** 2 for i, g in enumerate(_gamma[j])]) / sum(_gamma[j])
                self._alpha[j] = sum(_gamma[j]) / self.n
            if self._convergence(): break
        
        return {
            'mu': self._mu,
            'sigma': self._sigma,
            'alpha': self._alpha
        }


y = [-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]
ada = EMGaussain(2, 0.5, [-10, 10], 10)
print(ada.fit_predict(y))
