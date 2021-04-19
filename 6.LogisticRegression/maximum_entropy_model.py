import math
from copy import deepcopy

# 改进的迭代尺度算法IIS
# 参考 https://blog.csdn.net/breeze_blows/article/details/86612834
class MaximumEntropy:
    _samples = []   # 样本集合
    _Y = set()      # 标签集合，相当去去重后的y
    _exp_eps = {}   # key为(x,y)，value为出现次数
    _N = 0          # 样本数
    _M = 0          # 特征总数
    _w = {}         #
    def __init__(self, EPS = 0.005):
        self._EPS = EPS     # 收敛条件

    def loadData(self, dataset):
        self._samples = deepcopy(dataset)
        self._N = len(self._samples)

        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._Y.add(y)  # 集合中y若已存在则会自动忽略
            for x in X:
                self._exp_eps.setdefault((x, y), 0)
                self._exp_eps[(x, y)] += 1

        self._M = max([len(sample) - 1 for sample in self._samples])

        self._Ep_ = [0] * len(self._exp_eps)
        for xy, times in self._exp_eps.items():   # 计算特征函数fi关于经验分布的期望
            self._exp_eps[xy] = times / self._N
            self._w[xy] = 0

    def _Zx(self, X):
        # 计算每个特征值的Z(x)值
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._exp_eps: 
                    ss += self._w[(x, y)]
            zx += math.exp(ss)
        return zx

    def _conditional_probability(self, y, X):
        # 计算每个P(y|x) 既该条件下识别为y的概率
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._exp_eps: ss += self._w[(x, y)]
        res = math.exp(ss) / zx
        return res
 
    def _condition_ep(self, x, y):
        # 计算特征函数fi关于模型的期望
        ep = 0
        for sample in self._samples:
            if x not in sample: continue
            conditional_probability = self._conditional_probability(y, sample)
            ep += conditional_probability / self._N
        return ep

    def _convergence(self, lastw):  # 判断是否全部收敛
        for last, now in zip(lastw.values(), self._w.values()):
            if abs(last - now) >= self._EPS: return False
        return True

    def predict(self, X): # 计算预测概率
        result = {}
        for y in self._Y:
            result[y] = self._conditional_probability(y, X)

        return result

    def train(self, maxiter = 1000):   # 训练数据
        for loop in range(maxiter):  # 最大训练次数
            lastw = self._w.copy()
            for (x, y), exp_ep in self._exp_eps.items():
                condition_ep = self._condition_ep_ep (x, y) # 计算第i个特征的模型期望
                self._w[(x, y)] += math.log(exp_ep / condition_ep) / self._M # 更新参数

            if self._convergence(lastw): break # 判断是否收敛


dataset = [
    ['no', 'sunny', 'hot', 'high', 'FALSE'],
    ['no', 'sunny', 'hot', 'high', 'TRUE'],
    ['yes', 'overcast', 'hot', 'high', 'FALSE'],
    ['yes', 'rainy', 'mild', 'high', 'FALSE'],
    ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
    ['no', 'rainy', 'cool', 'normal', 'TRUE'],
    ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
    ['no', 'sunny', 'mild', 'high', 'FALSE'],
    ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
    ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
    ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
    ['yes', 'overcast', 'mild', 'high', 'TRUE'],
    ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
    ['no', 'rainy', 'mild', 'high', 'TRUE']
]

maxent = MaximumEntropy()
x = ['overcast', 'mild', 'high', 'FALSE']
maxent.loadData(dataset)
maxent.train()
print('predict:', maxent.predict(x))
