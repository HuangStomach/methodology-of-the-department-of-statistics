
import numpy as np
from copy import deepcopy
from typing import TypeVar

# 条件随机场模型学习的改进的迭代尺度算法
P = TypeVar('P', bound='CRFIIS')
class CRFIIS:
    def __init__(self, times = 1, epsilon = 0.001) -> None:
        self._times = times
        self._epsilon = epsilon

crf = CRFIIS()
