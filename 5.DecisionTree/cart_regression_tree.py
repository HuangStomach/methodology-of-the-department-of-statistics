import numpy as np
from typing import TypeVar

P = TypeVar('P', bound='CartClassificationTree')
# 目前统计学习方法关于该生成树的构造只有一维特征值的概念
class CartClassificationTree:
    '''
    最小二乘回归树
    '''
    tree = []
    tree_temp = []
    tree_leaf_count = 0
    loss_threshold = 0.0
    def __init__(self, loss_threshold) -> None:
        '''
        loss_threshold: 平方误差阈值，当小于等于该阈值时停止构造决策树
        '''
        self.loss_threshold = loss_threshold
        pass

    def _square_error(self, left, mid, right) -> float:
        # 计算平方误差
        c1 = np.average(self.y[left:mid]) if len(self.y[left:mid]) > 0 else 0.0
        c2 = np.average(self.y[mid:right]) if len(self.y[mid:right]) > 0 else 0.0
        return sum([(tag - c1) ** 2 for tag in self.y[left:mid]]) + sum([(tag - c2) ** 2 for tag in self.y[mid:right]])
    
    def _square_lost(self) -> float:
        # 利用平方误差定义的损失函数
        sum = 0.0
        for i, tag in enumerate(self.y):
            sum += (tag - self._predict_once(self.X[i])) ** 2
        return sum

    def _split(self, left, right) -> None:
        if right - left <= 1:
            self.tree_temp.append([])
            self.tree_temp.append([])
            self.tree_leaf_count += 2

        # 计算出每个分组的平方误差
        error = []
        for i in range(len(self.X)):
            error.append(self._square_error(left, i, right))
            
        m = min(error)
        mid = error.index(m)

        # 取误差最小的分组形式进行分组
        c1 = np.average(self.y[left:mid + 1])
        c2 = np.average(self.y[mid:right])
        self.tree_temp.append([c1, 'lt' if left == 0 else 'bt', left, mid])
        self.tree_temp.append([c2, 'gt' if right == len(self.y) else 'bt', mid, right])

        pass

    def fit(self, X, y) -> P:
        self.X = X
        self.y = y
        # 构造哨兵节点和根结点
        self.tree.append(0)
        self.tree.append([-1, 'bt', 0, len(y)])

        i = 0
        while True:
            # 根据二叉树的规则对树进行分裂
            for j in range(2 ** i, 2 ** (i + 1)):
                if j >= len(self.y) : return

                self._split(self.tree[j][2], self.tree[j][3])
                self.tree.extend(self.tree_temp)
                self.tree_temp.clear()
                # 每次分组后计算平方误差是否符合条件，符合后直接终止
                if (self._square_lost() <= self.loss_threshold) : return

            # 如果本次添加的均为空节点，则终止生成
            if self.tree_leaf_count == 2 ** (i + 1) : return
            self.tree_leaf_count = 0
            i += 1
    
    def _cond_select(self, x, item):
        if len(item) == 0: return np.NaN

        expect = item[0]
        op = item[1]
        if op == 'lt':
            if x <= self.X[item[3]]:
                return expect
        elif op == 'gt':
            if x > self.X[item[2]]:
                return expect
        else :
            if x > self.X[item[2]] and x <= self.X[item[3]]:
                return expect
        return np.NaN

    def _predict_once(self, x) -> float:
        # 根据树的叶子节点来进行单一值的预测
        length = len(self.tree)
        expect = 0.0
        i = 1
        while True:
            if i * 2 >= length: return expect

            # 如果该节点的两个子节点均为空节点则使用父节点的预测值
            rei = self._cond_select(x, self.tree[i * 2])
            rej = self._cond_select(x, self.tree[i * 2 + 1])
            if np.isnan(rei) and np.isnan(rej) : return expect

            if np.isnan(rej) :
                expect = rei
                i = i * 2
            else :
                expect = rej
                i = i * 2 + 1


X_train = list(range(1, 11))
y_train = [4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00]

tree = CartClassificationTree(1)
tree.fit(X_train, y_train)
print(tree.tree)
