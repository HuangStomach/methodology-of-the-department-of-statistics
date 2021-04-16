from decision_tree import DecisionTree

class DecisionTreeId3(DecisionTree):
    def __init__(self, epsilon) -> None:
        DecisionTree.__init__(self, epsilon)
        pass

    def information_gain(self, X, y, ignore) -> list:
        '''
        计算信息增益
        X: 数据集
        y: 分类标记
        ignore: 忽略具体特征下标
        '''
        length = len(X)
        d = self._d(X, y, ignore)
        res = []
        
        exp_entropy = self._exp_entropy(length, y) # 总数据集经验熵
        for i in range(len(X[0])):
            if i in ignore:
                res.append(0.0)
                continue

            condition_entropy = 0.0 # 经验条件熵
            for f, item in d[i].items():
                condition_entropy += item[0] / length * self._exp_entropy(item[0], item[1])
            res.append(exp_entropy - condition_entropy)

        return res

    def fit(self, X, y):
        self.root = self._build(X, y)
        pass
    
    def pruning(self):
        '''
        TODO: 决策树剪枝
        '''
        pass

    def show(self):
        '''
        TODO: 打印决策树
        '''

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

decision = DecisionTreeId3()
decision.fit(X_train, y_train)