import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='Perceptron')
class Perceptron:
    b = 0
    eta = 1
    modifed = True
    def __init__(self, eta = 1) -> None:
        self.eta = eta

    def _gram_matrix(self) -> None:
        # 生成Gram矩阵，方便之后直接取数据无需过多计算
        self.gram = np.zeros((self.len, self.len))
        for i in range(self.len):
            for j in range(self.len):
                self.gram[i][j] = np.inner(self.X[i], self.X[j])
    
    def _omega(self):
        # 计算omega的累加值
        temp = np.zeros(self.X.shape[1])
        for i in range(self.len):
            temp += self.alpha[i] * self.y[i] * self.X[i]
        return temp

    def _fit_once(self) -> None:
        for i in range(self.len):
            # 样本点到超平面距离不为正，则处于错误的数据集中
            if y[i] * (np.inner(self._omega(), self.X[i]) + self.b) <= 0:
                # 误分类时调整平面的方向和截距 使超平面向该误分类的一侧移动
                self.modifed = True
                self.alpha[i] += self.eta
                self.b += self.eta * self.y[i]

    def fit(self, X, y) -> P:
        self.len = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(self.len)
        self._gram_matrix()
        
        # 如果有修改，则继续进行训练，并把该轮次训练状态设置为还未修改
        while self.modifed:
            self.modifed = False
            self._fit_once()
        return self

X = np.array([
    [3, 3],
    [4, 3],
    [1, 1]
])
y = np.array([1, 1, -1])

p = Perceptron()
p.fit(X, y)
print(p.alpha)
print(p.b)
