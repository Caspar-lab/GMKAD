import numpy as np
def objectfunction(yyK, graph_w):
    """
    Python版本的objectfunction函数
    使用向量化计算替代循环，提高效率
    计算目标函数：sum_{i,j} (K(i,i) - 2*K(i,j) + K(j,j)) * graph_w(i,j)
    """
    # 提取对角元素
    diagK = np.diag(yyK)
    
    # 使用广播计算 term1 和 term2 
    # 更高效的广播方式
    term1 = diagK.reshape(-1, 1)  # 列向量
    term2 = diagK.reshape(1, -1)  # 行向量
    
    # 计算目标矩阵: K(i,i) + K(j,j) - 2*K(i,j)
    objMat = term1 + term2 - 2 * yyK
    
    # 与图权重矩阵逐元素相乘并求和
    obj = np.sum(objMat * graph_w)
    
    return obj