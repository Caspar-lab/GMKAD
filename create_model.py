import numpy as np
# 8/23 22:31
from mkljknn_train_copy import mkljknn_train
# from mkljknn_train import mkljknn_train
def graphJKNN_parameter():
    """
    模拟MATLAB中的graphJKNN_parameter函数
    返回参数字典
    """
    """参数设置函数"""
    par = {}
    par['eps'] = 1e-3
    # par['gat'] = {'typ': 'linear_softmax'}
    par['gat'] = {'typ': 'linear_sigmoid'}
    par['eta'] = {'typ': 'linear_softmax'}
    par['ker'] = ['l', 'g0.1']
    par['loc'] = {'typ': 'linear'}
    par['nor'] = {
        'dat': ['true', 'true'],
        'ker': ['true', 'true'],
        'loc': 'true'
    }
    par['opt'] = 'libsvm'
    par['see'] = 42
    par['tau'] = 1e-3
    par['alptolerance'] = 1e-3
    return par

# 8/23 23:11
def create_model(train_data, train_labels, kernels, graph_weights):
# def create_model(train_data, train_labels, kernels, graph_weights, lam):
    """
    Python版本的create_model函数
    对应MATLAB的create_model函数
    """
    # 创建training结构体（用字典模拟）
    training = {
        'ind': np.arange(1, train_data.shape[0] + 1).reshape(-1, 1),  # 1-based索引
        'X': train_data,
        'y': train_labels.reshape(-1, 1)  # 确保是列向量
    }
    
    # 核参数设置 - 创建training_data cell数组（用列表模拟）
    training_data = [None] * 3
    # 如果是3个视图的情况
    for i in range(3):
        training_data[i] = training
    
    # 获取参数
    parameters = graphJKNN_parameter()
    
    # 设置核参数
    parameters['ker'] = kernels
    
    # 设置归一化参数
    parameters['nor'] = {
        'dat': ['true', 'true', 'true'],
        'ker': ['true', 'true', 'true']
    }
    
    # # 调用训练函数
    # model, eta = mkljknn_train(training_data, parameters, graph_weights, lam)

    # 8/23 23:10
    # 调用训练函数
    model = mkljknn_train(training_data, parameters, graph_weights)
    
    return model