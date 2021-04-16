import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='KDTree')
class KDTree:
    k = 5
    d = 0
    class Node:
        def __init__(self, val, tag) -> None:
            self.val = val
            self.tag = tag
            pass

    def __init__(self, k = 5) -> None:
        '''
        k: 选取几个近邻进行判断
        '''
        self.k = k

    def _build(self, left, right, deep):
        if left > right:
            return None
        # 按照该划分的坐标轴对数据进行部分排序
        l = deep % self.d
        self.X[left:right + 1] = sorted(self.X[left:right + 1], key = lambda x : x[0][l])
        
        mid = (left + right + 1) // 2
        node = self.Node(self.X[mid][0], self.X[mid][1])
        node.left = self._build(left, mid - 1, deep + 1)
        node.right = self._build(mid + 1, right, deep + 1)

        return node

    def fit(self, X, y) -> P:
        # 构造kd树
        self.X = [];
        self.d = len(X[0])
        for i, x in enumerate(X):
            self.X.append((x, y[i]))
            
        # 此时self.X中的形式为元祖：（坐标，标签）
        self.X = sorted(self.X, key = lambda x : x[0][0])

        # kd树的根结点，递归生成
        self.root = self._build(0, len(self.X) - 1, 0)
        return self
    
    '''
    TODO: 缺少预测方法，不过网上大部分对具体预测很模糊
    由于需要从根结点回溯，虽然树结构也可以满足，但也许数组形式更容易完成
    另外需要排序数组来记录最近k个近邻，感觉消耗有点大，待实现
    '''

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

kdtree = KDTree(7)
#kdtree.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
