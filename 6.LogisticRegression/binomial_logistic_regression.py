from math import exp
import numpy as np
import matplotlib.pyplot as plt

# 逻辑斯谛二项回归模型
# 参考 https://blog.csdn.net/breeze_blows/article/details/86612834
class BinomialLogisticRegression:
    times = 200
    alpha = 0.01
    def __init__(self, times = 200, alpha = 0.01):
        self.max_iter = times # 最大迭代次数
        self.alpha = alpha # 学习梯度

    def sigmoid(self, x):
        return 1 / (1 + exp(-x)) # 使样本符合逻辑斯谛分布

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([*d, 1.0]) # 在每个样本特征后面加入1.0，扩充计算偏置
        return data_mat

    def fit(self, X, y):
        data_mat = self.data_matrix(X)
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32) # 初始化权重
        
        for iter_ in range(self.times):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights)) # 该分类的概率
                error = y[i] - result # 误差
                # 梯度上升法的参数更新过程：
                # thet := thet - alpha * ∂J(thet)/∂thet 对thet做最大似然估计
                self.weights += self.alpha * error * np.transpose([data_mat[i]]) # 对数似然函数求导后的算式 梯度上升法
    
    def predict(self, X_test) -> list:
        res = []
        X_test = self.data_matrix(X_test)
        for x in X_test:
            result = np.dot(x, self.weights)
            res.append(1 if result > 0 else 0)
        return np.array(res)

    def score(self, X_test, y_test):
        right = 0
        for i, y in enumerate(self.predict(X_test)):
            if (y == y_test[i]): right += 1
        return right / len(X_test)

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'][:100], iris_dataset['target'][:100], random_state=0
)

lr_clf = BinomialLogisticRegression()
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_test, y_test))
