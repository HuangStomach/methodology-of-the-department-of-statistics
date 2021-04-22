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
        self.gram = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.gram[i][j] = np.inner(self.X[i], self.X[j])
    
    def _omega(self):
        # 计算omega的累加值
        temp = np.zeros(self.X.shape[1])
        for i in range(self.n):
            temp += self.alpha[i] * self.y[i] * self.X[i]
        return temp

    def _fit_once(self) -> None:
        for i in range(self.n):
            # 样本点到超平面距离不为正，则处于错误的数据集中
            if self.y[i] * (np.inner(self._omega(), self.X[i]) + self.b) <= 0:
                # 误分类时调整平面的方向和截距 使超平面向该误分类的一侧移动
                self.modifed = True
                self.alpha[i] += self.eta
                self.b += self.eta * self.y[i]

    def fit(self, X, y) -> P:
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(self.n)
        self._gram_matrix()
        
        # 如果有修改，则继续进行训练，并把该轮次训练状态设置为还未修改
        while self.modifed:
            self.modifed = False
            self._fit_once()
        
        self.w = np.zeros(len(X[0]))
        for i in range(self.n):
            self.w += self.alpha[i] * y[i] * np.array(X[i])

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
