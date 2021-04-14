from decision_tree import DecisionTree

class DecisionTreeId3(DecisionTree):
    def __init__(self) -> None:
        pass

    def _build(self, X, y, ignore = set()):
        '''
        构建决策树
        ignore: 递归中已经处理完的特征值，进行忽略
        '''
        information_gain = self.information_gain(X, y, ignore)
        print(information_gain)
        index = information_gain.index(max(information_gain))
        ignore.add(index)
        node = self.Node(index, 0)

        features = {} # 计算在该特征下，分类情况
        for i, x in enumerate(X):
            features.setdefault(x[index], set())
            features[x[index]].add(y[i])
        
        for feature, tags in features.items():
            # 该特征下的分类均归一，则为叶节点
            
            if (len(tags) == 1):
                node.children.append(self.Node(index, tags.pop()))
                continue

            # 否则进行下一个特征决策 过滤出只符合该特征的数据子集
            X_new = [x for x in X if x[index] == feature]
            y_new = [y for i, y in enumerate(y) if X[i][index] == feature]
            node.children.append(self._build(X_new, y_new, ignore))

        return node

    def fit(self, X, y):
        self.root = self._build(X, y)
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