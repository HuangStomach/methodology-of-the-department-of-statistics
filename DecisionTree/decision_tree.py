import abc
import math
from typing import TypeVar

P = TypeVar('P', bound='DecisionTree')
class DecisionTree(metaclass=abc.ABCMeta):
    '''
    决策树基类，提供接口计算熵
    '''
    d = []
    class Node:
        def __init__(self, val, tag) -> None:
            self.val = val
            self.tag = tag
            self.children = []
            pass

    def __init__(self) -> None:
        pass

    def exp_entropy(self, length, y) -> float:
        '''
        计算某一特征在数据集下的熵
        length: 数据集长度
        y: 分类标记
        '''
        tags = {} # 标签： 训练样本下属于该标签的样本数量
        for tag in y:
            tags.setdefault(tag, 0)
            tags[tag] += 1 # 记录每个特征出现的次数

        exp_entropy = 0.0
        # 累加每个特征的熵
        for tag, cnt in tags.items():
            exp_entropy -= (cnt / length) * math.log(cnt / length, 2)

        return exp_entropy

    def _d(self, X, y, ignore) -> dict:
        '''
        d: [
            { feature: [num, [tag]], feature: [num, [tag]] } 特征值：(训练样本下该特征出现的次数, [该样本下出现的标签]) 
        ]
        '''
        res = []
        for i in range(len(X[0])):
            if i in ignore:
                res.append({})
                continue

            d = {}
            for x, tag in zip(X, y):
                f = x[i] # 特征值
                d.setdefault(f, [0, []])
                d[f][0] += 1
                d[f][1].append(tag)
            res.append(d)
        return res


    def information_gain(self, X, y, ignore) -> float:
        '''
        计算信息增益
        X: 数据集
        y: 分类标记
        i: 具体特征下标
        '''
        length = len(X)
        d = self._d(X, y, ignore)
        res = []
        
        exp_entropy = self.exp_entropy(length, y) # 总数据集经验熵
        for i in range(len(X[0])):
            if i in ignore:
                res.append(0.0)
                continue

            condition_entropy = 0.0 # 经验条件熵
            for f, item in d[i].items():
                condition_entropy += item[0] / length * self.exp_entropy(item[0], item[1])
            res.append(exp_entropy - condition_entropy)

        return res

    @abc.abstractmethod
    def fit(self, X, y) -> P:
        pass
