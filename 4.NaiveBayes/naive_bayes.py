import numpy as np
from typing import TypeVar

from sklearn.utils.validation import FLOAT_DTYPES

P = TypeVar('P', bound='Naive_Bayes')
class Naive_Bayes:
    _p = {}
    _conditions = {}
    d = 0 # 维度
    def __init__(self, lambd = 0) -> None:
        '''
        lambd: 默认值为0，调整贝叶斯估计平滑参数
        '''
        self.lambd = lambd

    def _prior(self, y) -> None:
        for tag in y:
            self._p.setdefault(tag, 0)
            self._conditions.setdefault(tag, {})
            self._p[tag] += 1

    def fit(self, X, y) -> P:
        self.d = len(X[0])
        self.l = len(X)

        # 初始化先验概率
        self._prior(y)
        self.k = len(self._p)

        # 按照各个维度来进行排序，构造条件概率
        for i in range(self.d):
            for x, tag in zip(X, y):
                self._conditions[tag].setdefault(x[i], 0)
                self._conditions[tag][x[i]] += 1

        return self

    def _predict_one(self, x) -> int:
        # 计算距离并取最近K个距离进行分类识别
        _tag = 0
        _max = 0.0
        for tag, num in self._p.items():
            # 计算先验概率
            chance = (num + self.lambd) / (self.l + self.k * self.lambd)
            # 针对每个维度计算条件概率
            for i in range(self.d):
                self._conditions[tag].setdefault(x[i], 0)
                chance *= (self._conditions[tag][x[i]] + self.lambd) / (num + self.d * self.lambd)
            if chance > _max:
                _max = chance
                _tag = tag

        return _tag
    
    def predict(self, X) -> list:
        res = []
        for x in X:
            res.append(self._predict_one(x))
        return np.array(res)

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

nb = Naive_Bayes(1)
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print(y_pred)
print(y_test)
