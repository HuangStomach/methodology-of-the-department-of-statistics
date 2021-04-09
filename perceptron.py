import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='Perceptron')
class Perceptron:
    w = np.array([0, 0])
    b = 0
    modifed = True
    def __init__(self) -> None:
        pass

    def _fit_once(self, X, y) -> None:
        for i in range(X.shape[0]):
            # 样本点到超平面距离不为正，则处于错误的数据集中
            if y[i] * (np.inner(self.w, X[i]) + self.b) <= 0:
                # 误分类时调整平面的方向和截距 使超平面向该误分类的一侧移动
                self.modifed = True
                self.w += X[i] * y[i]
                self.b += y[i]

    def fit(self, X, y) -> P:
        # 如果有修改，则继续进行训练，并把该轮次训练状态设置为还未修改
        while self.modifed:
            self.modifed = False
            self._fit_once(X, y)
        return self

X = np.array([
    [3, 3],
    [4, 3],
    [1, 1]
])
y = np.array([1, 1, -1])

p = Perceptron()
p.fit(X, y)
print(p.w)
print(p.b)