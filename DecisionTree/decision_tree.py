import abc
import math
import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='DecisionTree')
class DecisionTree(metaclass=abc.ABCMeta):
    '''
    决策树基类，提供接口计算熵
    leaf_count: 叶节点个数
    epsilon: 信息增益阈值
    '''
    d = []
    leaf_count = 0
    epsilon = 0.0
    class Node:
        def __init__(self, val, tag) -> None:
            '''
            val: 该节点对应特征的索引
            tag: 该节点归属的分类标记
            children: 子节点集合
            y: 该节点下分类标记集合
            '''
            self.val = val
            self.tag = tag
            self.children = []
            self.y = []
            pass

    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon
        pass

    def _exp_entropy(self, length, y) -> float:
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

    def _build(self, X, y, ignore = set()):
        '''
        构建决策树
        ignore: 递归中已经处理完的特征值，进行忽略
        '''
        info = self.information_gain(X, y, ignore)
        index = info.index(max(info))
        ignore.add(index)
        node = self.Node(index, 0) # 暂定0不作为标签使用
        node.y = y

        if max(info) < self.epsilon: # 当信息增益小于阈值的时候，将X集合中实例数最大的类设置为类标记, 直接设置为叶节点
            counts = np.bincount(y)
            node.tag = np.argmax(counts)
            self.leaf_count += 1
            return node

        features = {} # 计算在该特征下，分类情况
        for i, x in enumerate(X):
            features.setdefault(x[index], set())
            features[x[index]].add(y[i])
        
        for feature, tags in features.items():
            # 该特征下的分类均归一，则为叶节点
            
            if (len(tags) == 1):
                node.children.append(self.Node(index, tags.pop()))
                self.leaf_count += 1
                continue

            # 否则进行下一个特征决策 过滤出只符合该特征的数据子集
            X_new = [x for x in X if x[index] == feature]
            y_new = [y for i, y in enumerate(y) if X[i][index] == feature]
            node.children.append(self._build(X_new, y_new, ignore))

        return node

    @abc.abstractmethod
    def information_gain(self, X, y, ignore) -> list:
        '''
        计算信息增益相关取值
        X: 数据集
        y: 分类标记
        ignore: 忽略具体特征下标
        '''
        pass

    @abc.abstractmethod
    def fit(self, X, y) -> P:
        pass
