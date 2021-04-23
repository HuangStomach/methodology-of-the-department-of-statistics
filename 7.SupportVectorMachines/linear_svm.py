import numpy as np
import matplotlib.pyplot as plt

# 线性可分支持向量机
class LinearSVM:
    times = 200
    C = 1.0
    b = 0.0

    def __init__(self, times = 200, C = 1.0):
        self.times = times # 最大迭代次数
        self.C = C # 松弛变量

    def _violation_alpha(self):
        # 遍历样本, 检验是否满足KKT
        for i in range(self.n):
            if self._KKT(i): continue

            error = self.error[i]
            # 如果error是+，选择最小的；如果error是负的，选择最大的
            j = self.error.index(min(self.error) if error >= 0 else max(self.error))
            return i, j # 返回两个错误的索引
    
    # 决策函数 Σαi*yi*dot(x, xi) + b
    def decision_function(self, x):
        r = self.b
        for j in range(self.n):
            r += self.alpha[j] * self.y[j] * self.kernel(x, self.X[j])
        return r

    def _KKT(self, i): 
        # KKT 互补对偶条件
        st = self.decision_function(self.X[i]) * self.y[i] # 约束条件 
        if self.alpha[i] == 0 and st >= 1: return True
        elif 0 < self.alpha[i] < self.C and st == 1: return True
        elif self.alpha[i] == self.C and st <= 1: return True
        return False
    
    # 核方法
    def kernel(self, x1, x2):
        return np.dot(x1, x2)

    def fit(self, X, y):
        self.m = len(X[0])
        self.n = len(X)
        self.X = X
        self.y = y

        self.alpha = np.zeros(self.n) # 拉格朗日乘子
        self.error = [(self.decision_function(X[i]) - y[i]) for i in range(self.n)] # 存储误差

        for _i in range(self.times):
            i1, i2 = self._violation_alpha()

            # 边界 其中L与H是alpha所在对角线断点的界
            if y[i1] == y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            error1 = self.error[i1]
            error2 = self.error[i2]
            # eta = K11 + K22 - K12
            K11 = self.kernel(X[i1], X[i1])
            K22 = self.kernel(X[i2], X[i2])
            K12 = self.kernel(X[i1], X[i2])
            K21 = self.kernel(X[i2], X[i1])
            eta = K11 + K22 - 2 * K12

            alpha2_new_unc = self.alpha[i2] + y[i2] * (error1 - error2) / eta # 沿着约束方向未经剪辑时α2的最优解

            alpha2_new = alpha2_new_unc
            if alpha2_new_unc > H: alpha2_new = H
            elif alpha2_new_unc < L: alpha2_new = L

            # α1new = α1old + y1y2(α2old - α2new)
            alpha1_new = self.alpha[i1] + y[i1] * y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = - error1 - y[i1] * K11 * (alpha1_new - self.alpha[i1]) - y[i2] * K21 * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = - error2 - y[i1] * K12 * (alpha1_new - self.alpha[i1]) - y[i2] * K22 * (alpha2_new - self.alpha[i2]) + self.b

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            if 0 < alpha1_new < self.C: self.b = b1_new
            elif 0 < alpha2_new < self.C: self.b = b2_new
            else: self.b = (b1_new + b2_new) / 2 # 选择中点

            self.error[i1] = self.decision_function(X[i1]) - y[i1]
            self.error[i2] = self.decision_function(X[i2]) - y[i2]

        return self

    def predict(self, X_test) -> list:
        res = []
        for x in X_test:
            res.append(np.sign(self.decision_function(x)))
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

svm = LinearSVM()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))
