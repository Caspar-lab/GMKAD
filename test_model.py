import numpy as np
from typing import Tuple, Dict, Any, List, Union
from GMKAD import GMKAD

def test_model(test_data: np.ndarray, model: Dict, 
               JNN: int, centers: np.ndarray, index: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python版本的test_model函数
    使用训练好的模型对测试数据进行预测
    
    参数:
        test_data: 测试数据矩阵 (N_test x D)
        model: 训练好的模型字典
        kernels: 核参数列表，如 ['g0.1', 'p2']
        JNN: 近邻数参数
        KNN: 近邻数参数  
        number: 测试样本数量
        
    返回:
        predicted_labels: 预测标签 (N_test,)
        outlier_scores: 异常分数 (N_test,)
    """
    # 创建测试数据结构
    test_dict = {
        'X': test_data
    }
    
    # 创建测试数据列表（3个视图）
    test_data_list = [test_dict] * 3
    # 如果是4个视图的情况
    # test_data_list = [test_dict] * 4
    
    # 调用测试函数 - 根据MATLAB代码，当前使用的是Copy_3_of_GBMAD_test
    outlier_scores = GMKAD(test_data_list, model, JNN)

    # outlier_scores = MNOF_GB(test_data_list, model, JNN, centers, index, weights)

    # outlier_scores = MNOF(test_data_list, model, JNN)
    
    return outlier_scores