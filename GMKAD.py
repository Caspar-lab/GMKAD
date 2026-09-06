import numpy as np
import scipy.linalg as la
from scipy.sparse import diags
from typing import Dict, Any, Tuple
from scipy.spatial.distance import cdist

def weight_G_kdtree(data, dis_knn):
    """
    使用KD树加速的图权重计算
    """
    from sklearn.neighbors import NearestNeighbors
    
    num_data = data.shape[0]
    
    # 使用KD树快速找到k近邻
    # nbrs = NearestNeighbors(n_neighbors=dis_knn+1, algorithm='auto').fit(data)
    # distances, indices = nbrs.kneighbors(data)
    
    # 排除自身（第一个最近邻是自己）
    # knn_indices = indices[:, 1:]  # 形状: (num_data, dis_knn)
    
    # 初始化图矩阵
    Graph = np.ones((num_data, num_data))
    
    # # 预计算所有点对的内积
    # dot_products = data @ data.T  # 矩阵乘法计算所有内积
    
    # # 设置k近邻的权重
    # for i in range(num_data):
    #     for j in range(dis_knn):
    #         neighbor_idx = knn_indices[i, j]
    #         dot_val = dot_products[i, neighbor_idx]
    #         Graph[i, neighbor_idx] = (dot_val + 1) ** 2
    
    return Graph

def kernel(tra1, tra2, ker_type, nor_ker):
    """核函数计算"""
    X1 = tra1['X'] if isinstance(tra1, dict) else tra1
    X2 = tra2['X'] if isinstance(tra2, dict) else tra2
    
    if ker_type.startswith('g'):  # 高斯核
        sigma = float(ker_type[1:])
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        
        # 计算平方欧氏距离矩阵
        sq_dists = cdist(X1, X2, 'sqeuclidean')
        
        # 计算高斯核
        # K = np.exp(-sq_dists / (2 * sigma))
        K = np.exp(-sq_dists / (sigma ** 2))
        
        return K
    
    elif ker_type.startswith('p'):  # 多项式核
        # degree = int(ker_type[1:])
        # K = (np.dot(X1, X2.T) + 1) ** degree
        # return K

        degree = int(ker_type[1:])
        eps = 1e-16
        
        K = (np.dot(X1, X2.T) + 1.0) ** degree
        
        if nor_ker:
            x_norm = (np.sum(X1**2, axis=1, keepdims=True) + 1.0) ** degree
            y_norm = (np.sum(X2**2, axis=1, keepdims=True) + 1.0) ** degree
            K = K / (np.sqrt(x_norm @ y_norm.T) + eps)
        
        return K
    
    else:  # 线性核
        return np.dot(X1, X2.T)

def kernel_eta_vectorized(Km, eta):
    """
    向量化优化的kernel_eta函数
    
    使用广播和向量化操作替代循环，大幅提升性能
    """
    # 使用einsum进行高效计算
    # 公式: Keta = sum_m (eta[:,m] * eta[:,m]^T * Km[:,:,m])
    # einsum表示: 'im,jm,ijm->ij'
    Keta = np.einsum('im,jm,ijm->ij', eta, eta, Km, optimize=True)
    
    return Keta


def etas_robust(X, gat, eps, typ, safe_eps=1e-15):
    """
    更健壮的etas函数，包含数值稳定性处理
    
    参数:
        X: 输入数据
        gat: 门控权重矩阵
        eps: 阈值参数
        typ: 门控类型
        safe_eps: 数值稳定性小常数
        
    返回:
        eta: 多核权重矩阵
    """
    # 输入验证
    if not isinstance(X, np.ndarray) or not isinstance(gat, np.ndarray):
        raise ValueError("X和gat必须是numpy数组")
    
    if X.ndim != 2 or gat.ndim != 2:
        raise ValueError("X和gat必须是二维数组")
    
    N, DG = X.shape
    P, DG_plus_1 = gat.shape
    
    if DG_plus_1 != DG + 1:
        raise ValueError(f"gat的列数应为{X.shape[1] + 1}，但实际为{DG_plus_1}")
    
    typ = typ.lower()
    
    try:
        if typ in ['constant_softmax']:
            # 常数softmax
            constant_features = np.hstack([np.ones((N, 1)), np.zeros_like(X)])
            val = constant_features @ gat.T
            # 数值稳定的softmax
            val = val - np.max(val, axis=1, keepdims=True)
            val = np.clip(val, -500, 500)  # 防止指数溢出
            eta = np.exp(val)
            eta = eta / (np.sum(eta, axis=1, keepdims=True) + safe_eps)
            
        elif typ in ['constant_sigmoid']:
            # 常数sigmoid
            constant_features = np.hstack([np.ones((N, 1)), np.zeros_like(X)])
            val = -constant_features @ gat.T
            val = np.clip(val, -500, 500)  # 防止指数溢出
            eta = 1.0 / (1.0 + np.exp(val))
            
        elif typ in ['linear_sigmoid', 'sigmoid']:
            # 线性sigmoid
            X_aug = np.hstack([np.ones((N, 1)), X])
            val = -X_aug @ gat.T
            val = np.clip(val, -500, 500)
            eta = 1.0 / (1.0 + np.exp(val))
            
        elif typ in ['linear_softmax', 'softmax']:
            # 线性softmax
            X_aug = np.hstack([np.ones((N, 1)), X])
            val = X_aug @ gat.T
            val = val - np.max(val, axis=1, keepdims=True)
            val = np.clip(val, -500, 500)
            eta = np.exp(val)
            eta = eta / (np.sum(eta, axis=1, keepdims=True) + safe_eps)
            
        elif typ == 'rbf_softmax':
            # RBF softmax
            P = gat.shape[0]
            val = np.zeros((N, P))
            for m in range(P):
                # 使用广播计算距离
                diff = X - gat[m, 1:]
                # 防止带宽为0
                bandwidth_sq = max(gat[m, 0]**2, safe_eps)
                val[:, m] = -np.sum(diff**2, axis=1) / bandwidth_sq
            # softmax计算
            val = val - np.max(val, axis=1, keepdims=True)
            val = np.clip(val, -500, 500)
            eta = np.exp(val)
            eta = eta / (np.sum(eta, axis=1, keepdims=True) + safe_eps)
            
        else:
            # 默认使用线性softmax
            print(f"警告: 未知的门控类型 '{typ}'，使用线性softmax")
            X_aug = np.hstack([np.ones((N, 1)), X])
            val = X_aug @ gat.T
            val = val - np.max(val, axis=1, keepdims=True)
            val = np.clip(val, -500, 500)
            eta = np.exp(val)
            eta = eta / (np.sum(eta, axis=1, keepdims=True) + safe_eps)
        
        # 截断处理
        eta = np.clip(eta, 0, 1)  # 先确保在[0,1]范围内
        eta[eta < eps] = 0
        eta[eta > 1 - eps] = 1
        
        return eta
        
    except Exception as e:
        print(f"etas计算错误: {e}")
        # 返回默认值
        return np.ones((N, P)) / P

def GMKAD(tes, mod, dis_knn):
    """
    Python版本的Copy_3_of_GBMAD_test函数
    基于优化后的多核系数构建多核距离矩阵，然后在多核距离矩阵的基础上进行随机游走AD
    
    参数:
        tes: 测试数据列表，包含多个视图
        mod: 训练好的模型
        dis_knn: 近邻数参数
        
    返回:
        out: 包含预测结果和异常分数的字典
    """
    d = 0.1  # 阻尼因子
    N = tes[0]['X'].shape[0]  # 测试样本数
    
    P = len(tes) - 1  # 视图数量

    # 用全体样本训练和测试可用
    # K_tete = mod['yyKeta']
        
    # 初始化核矩阵
    K_tete = np.zeros((N, N))

    loc = tes[0]['X']  # 局部性特征
    
    yyKm = np.zeros((N, N, P))
    # 计算多核组合
    for m in range(P):
        yyKm[:, :, m] = kernel(tes[m], tes[m], mod['par']['ker'][m], mod['par']['ker'][m])
        # 计算第m个核的核矩阵

    opt_gat = mod['gat_opt']
    par = mod['par']
    eps = par.get('eps', 1e-3)
    # 3. 计算核权重eta
    eta = etas_robust(loc, opt_gat, eps, par.get('gat', {}).get('typ', 'linear_softmax'))
    #  8/26
    # eta = etas_robust(loc, opt_gat, eps, par.get('eta', {}).get('typ', 'softmax'))
    
    # # 组合多核
    K_tete = kernel_eta_vectorized(yyKm, eta)
    
    # 平均化核矩阵
    K_tete = K_tete / P
    
    # 计算图权重
    WGraph = weight_G_kdtree(tes[0]['X'], dis_knn)
    
    # 计算基于核的距离矩阵
    if is_symmetric(K_tete):
        # 对称矩阵的高效计算
        diag_K = np.diag(K_tete)
        # 计算下三角部分（不包括对角线）
        temp = np.sqrt(np.maximum(diag_K[:, None] - 2 * np.tril(K_tete, -1) + diag_K, 0))
        # 将下三角部分复制到上三角部分，形成完整矩阵
        dis_Samples = temp + temp.T - np.diag(np.diag(temp))
    else:
        # 非对称矩阵的计算
        diag_K = np.diag(K_tete)
        dis_Samples = np.sqrt(np.maximum(diag_K[:, None] - 2 * K_tete + diag_K, 0))
    
    # 结合图权重的加权相似度
    weighted_similarities = WGraph * dis_Samples
    
    # 使用高斯核将距离转换为相似度
    sigma = 1.0  # 可调整参数
    A = np.exp(-weighted_similarities / (2 * sigma))
    
    # 将对角线元素设置为0
    np.fill_diagonal(A, 0)
    
    # 计算度矩阵（行和）
    diag_A = np.sum(A, axis=1)
    diag_A[diag_A == 0] = 1  # 防除零
    P = A / diag_A[:, np.newaxis]  # 广播行除法，等价于 B^{-1} @ A
    
    # 随机游走过程
    phi_t = np.ones(N) / N  # 初始概率分布
    phi_t_temp = np.ones(N)
    
    max_iter = 1000  # 最大迭代次数
    tolerance = 1e-4
    i = 0
    
    # 迭代直到收敛
    while np.linalg.norm(phi_t_temp - phi_t, 1) > tolerance and i < max_iter:
        phi_t_temp = phi_t.copy()
        phi_t = d + (1 - d) * (phi_t @ P)
        i += 1
    
    phi_t_w = phi_t
    
    # Convert the unnormalized stationary visit score to the manuscript's [0, 1] anomaly score.
    AD = phi_t_w
    score_range = np.max(AD) - np.min(AD)
    if score_range > 0:
        outlier_score = (np.max(AD) - AD) / score_range
    else:
        outlier_score = np.zeros_like(AD)
    
    return outlier_score

def is_symmetric(matrix, rtol=1e-5, atol=1e-8):
    """检查矩阵是否对称"""
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

def min_max_normalize(data):
    """最大最小归一化到[0,1]区间"""
    if data.size == 0:
        return data
    
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # 处理常数列
    range_vals = max_vals - min_vals
    mask = range_vals == 0
    range_vals[mask] = 1  # 避免除零
    
    normalized = (data - min_vals) / range_vals
    return normalized
