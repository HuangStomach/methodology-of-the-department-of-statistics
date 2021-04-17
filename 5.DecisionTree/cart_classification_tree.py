# TODO: 实现
import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='CartClassificationTree')
# 目前统计学习方法关于该生成树的构造只有一维特征值的概念
class CartClassificationTree:
    tree = [0, 'root']
    def __init__(self) -> None:
        pass

    def _gini(self, y) -> float:
        # 计算某个集合的基尼指数
        tags = np.unique(y)
        gini = 1.0
        for tag in tags:
            gini -= (y.count(tag) / len(y)) ** 2
        return gini

    def _gini_condition(self, X, y, i) -> dict:
        # 计算在特征i的条件下，集合的基尼指数
        res = {}

        features = np.unique(X[:, i])
        for feature in features:
            # 集合D1 符合该可能取值
            X_eq = [x for x in X if x[i] == feature]
            y_eq = [tag for index, tag in enumerate(y) if X[index][i] == feature]
            # 集合D2 不符合该可能取值
            X_else = [x for x in X if x[i] != feature]
            y_else = [tag for index, tag in enumerate(y) if X[index][i] != feature]
            
            res[feature] = len(X_eq) / len(X) * self._gini(y_eq) + len(X_else) / len(X) * self._gini(y_else)
        return res
    
    def _min_gini(self, gini_f) -> tuple:
        # gini_f 每个特征值下的基尼指数
        m = 1
        index = -1
        f = 0
        for i, feature in enumerate(gini_f):
            for val, gini in feature.items():
                if gini >= m: continue

                m = gini
                index = i
                f = val
        # 返回对应的特征值的索引和符合该最小基尼指数的特征值
        return (index, f)

    def _loss_function(self) -> float:

        pass

    def _tags_count(self, X, y, index, feature, op = 'eq'):
        # 样本集合在某个特征值取特定值时的分类数量
        y_new = []
        if op == 'eq': y_new = [y for i, y in enumerate(y) if X[i][index] == feature]
        else : y_new = [y for i, y in enumerate(y) if X[i][index] != feature]
        return y_new

    def _add_node(self, index, node) -> None:
        # 给数组动态扩容
        if index >= len(self.tree):
            self.tree.extend(np.zeros(index - len(self.tree) + 1))
        # 节点结构:[
        #   条件: 'eq,ne'
        #   特征值的索引
        #   特征值
        #   该条件下分类的集合
        # ]
        self.tree[index] = node
        
    def _build(self, X, y, father, ignore = set()) -> None:
        # 计算每一个特征值的基尼指数
        gini_f = []
        for i in range(len(X[0])):
            if i in ignore: gini_f.append({0: 1.0})
            else: gini_f.append(self._gini_condition(X, y, i))

        # 找出当前最小的基尼指数，并且计算出当前节点的索引
        index = 2 * father
        min_gini = self._min_gini(gini_f)
        ignore.add(min_gini[0])

        # 符合该特征条件的进行节点生成
        eq = self._tags_count(X, y, min_gini[0], min_gini[1])
        self._add_node(index, ('eq', min_gini[0], min_gini[1], eq))
        if len(np.unique(eq)) > 1: # 该条件下分类未归一，则筛选出数据子集继续分类
            X_new = np.array([x for x in X if x[min_gini[0]] == min_gini[1]])
            y_new = np.array([y for i, y in enumerate(y) if X[i][min_gini[0]] == min_gini[1]])
            self._build(X_new, y_new, index, ignore)
            
        ne = self._tags_count(X, y, min_gini[0], min_gini[1], 'ne')
        self._add_node(index + 1, ('ne', min_gini[0], min_gini[1], ne))
        if len(np.unique(ne)) > 1: # 该条件下分类未归一，则筛选出数据子集继续分类
            X_new = np.array([x for x in X if x[min_gini[0]] != min_gini[1]])
            y_new = np.array([y for i, y in enumerate(y) if X[i][min_gini[0]] != min_gini[1]])
            self._build(X_new, y_new, index + 1, ignore)

    def fit(self, X, y) -> None:
        if (type(X) is not np.ndarray):
            X = np.array(X)
        
        self._build(X, y, 1)
        print(self.tree)
    
    def pruning(self):
        '''
        TODO: 决策树剪枝
        '''
        k = 0
        alpha = float('inf')
        for node in self.tree:

            pass
        pass

X_train = [
    [0, 0, 0, 2],
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 2],
    [0, 0, 0, 2],
    [1, 0, 0, 2],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [2, 0, 1, 0],
    [2, 0, 1, 1],
    [2, 1, 0, 1],
    [2, 1, 0, 0],
    [2, 0, 0, 2],
]

y_train = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

decision = CartClassificationTree()
decision.fit(X_train, y_train)