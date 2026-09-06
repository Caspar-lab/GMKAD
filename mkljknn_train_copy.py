import numpy as np
from scipy.linalg import norm
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from objectfunction import objectfunction
from new_optimize_gat_adam_vector import new_optimize_gat_adam_vectorized
def print_optimization_summary(stats):
    """Print optimization process summary"""
    print("="*60)
    print("Optimization Process Summary")
    print("="*60)
    print(f"Iterations: {stats['iterations']}")
    print(f"Initial Objective: {stats['initial_obj']:.6f}")
    print(f"Final Objective: {stats['final_obj']:.6f}")
    print(f"Improvement Ratio: {stats['improvement']:.2f}%")
    print(f"Convergence Rate: {stats['convergence_rate']:.6f}")
    print(f"Gating Weight Mean: {stats['gat_mean']:.4f} ± {stats['gat_std']:.4f}")
    print(f"Gating Weight Sparsity: {stats['gat_sparsity']:.2%}")
    print(f"Multi-Kernel Weight Mean: {stats['eta_mean']:.4f} ± {stats['eta_std']:.4f}")
    print(f"Dominant Kernel Ratio: {stats['dominant_kernel_ratio']:.2%}")
    print("="*60)

def visualize_kernel_matrices(yyKm, max_display=4):
    """
    可视化核矩阵
    """
    N, N2, P = yyKm.shape
    
    # 限制显示的核数量
    display_P = min(P, max_display)
    
    fig, axes = plt.subplots(2, display_P, figsize=(4*display_P, 8))
    
    if display_P == 1:
        axes = [[axes[0]], [axes[1]]]
    
    for m in range(display_P):
        K = yyKm[:, :, m]
        
        # 热图
        sns.heatmap(K, ax=axes[0][m], cmap='viridis', 
                   cbar_kws={'label': f'核{m}值'})
        axes[0][m].set_title(f'核{m}热图')
        
        # 对角线值分布
        diag_values = np.diag(K)
        axes[1][m].hist(diag_values, bins=20, alpha=0.7)
        axes[1][m].axvline(1.0, color='red', linestyle='--', label='期望值=1.0')
        axes[1][m].set_xlabel('对角线值')
        axes[1][m].set_ylabel('频数')
        axes[1][m].set_title(f'核{m}对角线分布')
        axes[1][m].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计摘要
    print("核矩阵统计摘要:")
    for m in range(P):
        K = yyKm[:, :, m]
        diag_mean = np.mean(np.diag(K))
        print(f"核{m}: 对角线均值={diag_mean:.4f}, "
              f"范围=[{np.min(K):.2e}, {np.max(K):.2e}]")

def mean_and_std(data, normalize_type):
    """计算数据的均值和标准差"""
    if normalize_type == 'true':
        return {'mean': np.mean(data, axis=0), 'std': np.std(data, axis=0)}
    else:
        return {'mean': 0, 'std': 1}

def locality(data, loc_type):
    """计算局部性特征"""
    # 简化实现，返回原始数据或处理后的数据
    return data

import numpy as np

def gating_initial_robust(loc, P, typ):
    """
    更健壮的gating_initial函数，包含错误处理和验证
    
    参数:
        loc: 局部特征数据
        P: 核的数量
        typ: 门控类型
        
    返回:
        gat: 门控权重矩阵
    """
    # 输入验证
    if not isinstance(loc, np.ndarray):
        raise ValueError("loc必须是numpy数组")
    
    if loc.ndim != 2:
        raise ValueError("loc必须是二维数组")
    
    N, DG = loc.shape
    
    if P <= 0 or P > N:
        raise ValueError(f"P必须在1到{N}之间")
    
    # 根据类型初始化
    typ = typ.lower()  # 转换为小写以增强鲁棒性
    
    if typ in ['constant_sigmoid', 'linear_sigmoid', 'sigmoid', 
               'constant_softmax', 'linear_softmax', 'softmax']:
        # # 随机初始化，使用较小的随机值
        # np.random.seed(42)  # 可选的随机种子，确保可重复性
        # gat = np.random.rand(P, DG + 1)  # 缩小初始权重

        # # 生成随机数并按列优先填充
        total_elements = P * (DG + 1)
        random_flat = np.random.rand(total_elements)
        
        # 按列优先方式重塑（MATLAB风格）
        gat = random_flat.reshape((DG + 1, P)).T
    
    elif typ == 'rbf_softmax':
        # 从数据中随机选择P个样本作为中心
        np.random.seed(42)  # 可选的随机种子
        shu = np.random.permutation(N)
        if P > N:
            raise ValueError(f"P={P}不能大于样本数N={N}")
        
        gat = np.ones((P, DG + 1))
        gat[:, 1:] = loc[shu[:P], :].copy()  # 使用copy避免引用问题
    
    else:
        # 未知类型，使用默认的随机初始化
        print(f"警告: 未知的门控类型 '{typ}'，使用随机初始化")
        gat = np.random.rand(P, DG + 1)
    
    return gat

import numpy as np

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

def etas(loc_data, gat, eps, gat_type):
    """计算多核权重"""
    N = loc_data.shape[0]
    P = gat.shape[0]
    # 添加偏置项
    loc_with_bias = np.column_stack([loc_data, np.ones(N)])
    
    # 计算每个样本对每个核的权重
    eta = np.zeros((N, P))
    for i in range(N):
        for m in range(P):
            # 使用sigmoid函数
            z = np.dot(gat[m, :], loc_with_bias[i, :])
            eta[i, m] = 1.0 / (1.0 + np.exp(-z))
    
    # 归一化，使得每个样本的核权重和为1
    eta = eta / (np.sum(eta, axis=1, keepdims=True) + eps)
    return eta

def kernel(tra1, tra2, ker_type):
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
        K = np.exp(-sq_dists / (2 * sigma))
        # K = np.exp(-sq_dists / (sigma ** 2))
        
        return K
    
    elif ker_type.startswith('p'):  # 多项式核
        # degree = int(ker_type[1:])
        # K = (np.dot(X1, X2.T) + 1) ** degree
        # return K

        degree = int(ker_type[1:])
        eps = 1e-16
        
        # 分子: (x·y^T + 1)^q
        K = (np.dot(X1, X2.T) + 1.0) ** degree
        
        # 分母: sqrt( ((||x||^2+1)^q) * ((||y||^2+1)^q) ) + eps
        x_norm = (np.sum(X1**2, axis=1, keepdims=True) + 1.0) ** degree  # (n, 1)
        y_norm = (np.sum(X2**2, axis=1, keepdims=True) + 1.0) ** degree  # (m, 1)
        denom = np.sqrt(x_norm @ y_norm.T) + eps                         # (n, m)
        
        K = K / denom
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

def kernel_eta(yyKm, eta):
    """组合多核"""
    N, _, P = yyKm.shape
    yyKeta = np.zeros((N, N))
    
    for i in range(N):
        for m in range(P):
            # 使用eta权重组合核
            yyKeta += eta[i, m] * yyKm[:, :, m]
    
    return yyKeta

def mkljknn_train(tra, par, graph_w):
    """
    Python版本的mkljknn_train函数
    """
    # 设置随机种子
    if 'see' in par:
        np.random.seed(par['see'])
    
    P = len(tra) - 1
    mod = {}
    
    # 对每个视图进行归一化
    for m in range(P):
        scaler = MinMaxScaler()
        tra[m]['X'] = scaler.fit_transform(tra[m]['X']) # 使用min-max归一化
    
    # 计算局部性特征
    mod['loc'] = tra[P]['X']
    
    # 初始化门控权重
    mod['gat'] = gating_initial_robust(mod['loc'], P, par.get('gat', {}).get('typ', 'sigmoid'))
    
    # eps = par.get('eps', 1e-3)
    # eta = etas_robust(mod['loc'], mod['gat'], eps, par.get('gat', {}).get('typ', 'sigmoid'))
    # # 计算多核权重 8/26
    # # eta = etas_robust(mod['loc'], mod['gat'], eps, par.get('eta', {}).get('typ', 'softmax'))
    
    # N = tra[0]['X'].shape[0]
    # yyKm = np.zeros((N, N, P))
    
    # # 计算每个核的核矩阵
    # for m in range(P):
    #     yyKm[:, :, m] = kernel(tra[m], tra[m], par['ker'][m])
    
    # mod['yyKm'] = yyKm

    # # 组合多核
    # yyKeta = kernel_eta_vectorized(yyKm, eta)
    # mod['yyKold'] = yyKeta
    
    # # mod['yyKeta'] = yyKeta
    # # return mod, eta

    
    # # 计算目标函数
    # obj = objectfunction(yyKeta, graph_w)
    
    # # 训练过程
    # mod['obj'] = [obj]
    # mod['sol'] = 1
    
    # # 初始化门控参数
    # gat0 = mod['gat']
    
    # opts = {
    #     'lr': 1e-2,
    #     'max_iter': 500,
    #     'proj_interval': 5
    # }
    
    # # 多核情况下进行优化
    # if P > 1:
    #     oldObj = obj
        
    #     # 使用优化器（根据注释，最后使用的是new_optimize_gat_adam）
    #     start_time = time.time()
    #     gat_opt, obj_history, eta_opt = new_optimize_gat_adam_vectorized(tra[0]['X'], yyKm, graph_w, gat0, opts)
    #     print(f"Optimization time: {time.time() - start_time:.4f}s")
        
    #     # 执行可视化
    #     # stats = visualize_optimization_process(
    #     #     gat_opt=gat_opt,
    #     #     obj_history=obj_history,
    #     #     eta_opt=eta_opt,
    #     #     X=tra[0]['X'],
    #     #     show_plots=True
    #     # )
    #     # # 打印摘要
    #     # print_optimization_summary(stats)

    #     # 更新目标函数值
    #     obj = obj_history[-1]
    #     mod['obj'].append(obj)
        
    #     # 检查收敛条件（简化）
    #     # if abs(obj - oldObj) <= par.get('eps', 1e-8) * abs(oldObj):
    #     #     pass  # 收敛
    
    # # 保存优化结果
    # mod['gat_opt'] = gat_opt if P > 1 else mod['gat']
    # mod['eta_opt'] = eta_opt
    # mod['yyKeta'] = kernel_eta_vectorized(yyKm, eta_opt) if P > 1 else yyKeta
    
    # 保存支持信息
    # mod['sup'] = [None] * P
    # for m in range(P):
    #     mod['sup'][m] = {
    #         'ind': tra[m]['ind'],
    #         'X': tra[m]['X'],
    #         'y': tra[m]['y'],
    #         'eta': eta_opt[0, m] if m < eta.shape[1] else 1.0  # 简化
    #     }
    
    mod['par'] = par

    # 9/2
    mod['gat_opt'] = mod['gat']  # 如果没有优化，使用初始gat
    
    return mod

