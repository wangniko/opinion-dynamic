## 定义通用数学表达式
import numpy as np
## 将数据归一化到任意区间[a,b]的方法

def scaling(data, a, b):
    denominator = (np.max(data) - np.min(data))
    slope = (b - a) / denominator
    return np.dot(slope, data - np.min(data) ) + a

