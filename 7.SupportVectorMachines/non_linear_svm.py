import math
import numpy as np
import matplotlib.pyplot as plt

# 线性可分支持向量机
class LinearSVM:
    times = 200
    C = 1.0
    b = 0.0

    def __init__(self, times = 200, C = 1.0, kernel_function = 'polynomial', p = 2, sigma = 1.0):
        self.times = times # 最大迭代次数
        self.C = C # 松弛变量
        self.kernel_function = kernel_function # 核函数
        self.p = p # 多项式核函数的次数
        self.sigma = sigma # 高斯径向量分母

    def _violation_alpha(self):
        # 选择两个违反KKT条件的变量，固定其他变量，针对该两个变量进行二次规划
        # 遍历样本, 检验是否满足KKT

        # 先选择满足条件 0 < alpha < C也就是间隔边界上的支持向量
        for i in [i for i, alpha in enumerate(self.alpha) if alpha > 0 and alpha < self.C]:
            # 遍历样本, 检验是否满足KKT
            if self._KKT(i): continue
            return self._violation_alpha_select(i)

        # 如果没有再遍历全部结果集
        for i in [i for i, alpha in enumerate(self.alpha) if alpha == 0 or alpha == self.C]:
            # 遍历样本, 检验是否满足KKT
            if self._KKT(i): continue
            return self._violation_alpha_select(i)

    def _violation_alpha_select(self, i):
        error1 = self.error[i]
        # 如果error是+，选择最小的；如果error是负的，选择最大的
        j = self.error.index(min(self.error) if error1 >= 0 else max(self.error))
        alpha2_new = self._alpha2_new(i, j)
        
        if abs(self.alpha[j] - alpha2_new) < 0.000001:
            for j in [j for j, alpha in enumerate(self.alpha) if alpha > 0 and alpha < self.C]:
                alpha2_new = self._alpha2_new(i, j)
                if abs(self.alpha[j] - alpha2_new) < 0.000001: continue

        return i, j, alpha2_new # 返回两个错误的索引和相对有效的alpha2_new
    
    def _alpha2_new(self, i, j):
        error1 = self.error[i]
        error2 = self.error[j]

        # 边界 其中L与H是alpha所在对角线断点的界
        if self.y[i] == self.y[j]:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

        # eta = K11 + K22 - K12
        K11 = self.kernel(self.X[i], self.X[i])
        K22 = self.kernel(self.X[j], self.X[j])
        K12 = self.kernel(self.X[i], self.X[j])
        eta = K11 + K22 - 2 * K12
        if eta == 0: return self.alpha[j] # eta为0时不做选择

        alpha2_new_unc = self.alpha[j] + self.y[j] * (error1 - error2) / eta # 沿着约束方向未经剪辑时α2的最优解

        if alpha2_new_unc > H: alpha2_new = H
        elif alpha2_new_unc < L: alpha2_new = L
        else: alpha2_new = alpha2_new_unc

        return alpha2_new # 返回两个错误的索引和相对有效的alpha2_new

    def _KKT(self, i): 
        # KKT 互补对偶条件
        st = self.decision_function(self.X[i]) * self.y[i] # 约束条件 
        if self.alpha[i] == 0 and st >= 1: return True
        elif 0 < self.alpha[i] < self.C and st == 1: return True
        elif self.alpha[i] == self.C and st <= 1: return True
        return False
    
    # 决策函数 Σαi*yi*dot(x, xi) + b
    def decision_function(self, x):
        r = self.b
        for j in range(self.n):
            r += self.alpha[j] * self.y[j] * self.kernel(x, self.X[j])
        return r

    # 核函数
    def kernel(self, x1, x2):
        if self.kernel_function == 'gaussian':
            norm = - np.linalg.norm(np.array(x1) - np.array(x2)) ** 2
            return math.exp(norm / 2 * (self.sigma ** 2))
        
        # 默认视为使用多项式核函数，万一打错了给你纠正一下，不报错了
        self.kernel_function = 'polynomial'
        return (np.dot(x1, x2) + 1) ** self.p

    def fit(self, X, y):
        self.m = len(X[0])
        self.n = len(X)
        self.X = X
        self.y = y

        self.alpha = np.zeros(self.n) # 拉格朗日乘子
        self.error = [(self.decision_function(X[i]) - y[i]) for i in range(self.n)] # 存储误差

        for _i in range(self.times):
            i1, i2, alpha2_new = self._violation_alpha()
            if math.isnan(alpha2_new): break # 无法选出合适的 alpha2_new 也就是符合精度反胃了，

            # TODO: 待优化 存储 kernel
            error1 = self.error[i1]
            error2 = self.error[i2]
            # eta = K11 + K22 - K12
            K11 = self.kernel(X[i1], X[i1])
            K22 = self.kernel(X[i2], X[i2])
            K12 = self.kernel(X[i1], X[i2])
            K21 = self.kernel(X[i2], X[i1])

            # α1new = α1old + y1y2(α2old - α2new)
            alpha1_new = self.alpha[i1] + y[i1] * y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = - error1 - y[i1] * K11 * (alpha1_new - self.alpha[i1]) - y[i2] * K21 * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = - error2 - y[i1] * K12 * (alpha1_new - self.alpha[i1]) - y[i2] * K22 * (alpha2_new - self.alpha[i2]) + self.b

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            if 0 < alpha1_new < self.C and 0 < alpha2_new < self.C: self.b = b1_new
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
