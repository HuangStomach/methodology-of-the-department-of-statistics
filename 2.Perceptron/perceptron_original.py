import numpy as np
from typing import TypeVar

# TODO: 简化该类参数传递，并且似乎应该每次随机打乱训练数据
P = TypeVar('P', bound='Perceptron')
class Perceptron:
    b = 0
    modifed = True
    def __init__(self, eta = 1) -> None:
        self.eta = eta
        pass

    def _classify(self, x) -> int:
        return np.inner(self.w, x) + self.b

    def _fit_once(self, X, y) -> None:
        for i in range(X.shape[0]):
            # 样本点到超平面距离不为正，则处于错误的数据集中
            if y[i] * self._classify(X[i]) <= 0:
                # 误分类时调整平面的方向和截距 使超平面向该误分类的一侧移动
                self.modifed = True
                self.w += self.eta * X[i] * y[i]
                self.b += self.eta * y[i]

    def fit(self, X, y) -> P:
        self.w = np.zeros(X.shape[1])
        # 如果有修改，则继续进行训练，并把该轮次训练状态设置为还未修改
        while self.modifed:
            self.modifed = False
            self._fit_once(X, y)
        return self

    def predict(self, X_test) -> list:
        res = []
        for x in X_test:
            res.append(np.sign(np.dot(x, self.w) + self.b))
        return np.array(res)

    def score(self, X_test, y_test):
        right = 0
        for i, y in enumerate(self.predict(X_test)):
            if (y == y_test[i]): right += 1
        return right / len(X_test)

from sklearn.datasets import load_iris
iris_dataset = load_iris()
target = []
for y in iris_dataset['target']:
    if y == 0: target.append(-1)
    else: target.append(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'][:100], target[:100], random_state=0
)

p = Perceptron()
p.fit(X_train, y_train)
print(p.score(X_test, y_test))
