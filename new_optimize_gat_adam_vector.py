import numpy as np
from objectfunction import objectfunction
def new_optimize_gat_adam_vectorized(X, yyK, graph_w, gat0, opts):
    """
    更向量化的版本，性能更好
    """
    # 参数设置
    lr = opts.get('lr', 1e-3)
    beta1 = opts.get('beta1', 0.9)
    beta2 = opts.get('beta2', 0.999)
    eps = opts.get('eps', 1e-8)
    max_iter = opts.get('max_iter', 1500)
    tol = opts.get('tol', 1e-6)
    patience = opts.get('patience', 20)
    
    P, D = gat0.shape
    N = X.shape[0]
    gat = gat0.copy()
    m = np.zeros((P, D))
    v = np.zeros((P, D))
    X_aug = np.column_stack([np.ones(N), X])
    
    obj_history = []
    best_obj = float('inf')
    best_gat = gat.copy()
    patience_counter = 0

    # # 度量两个核的相似程度 核冗余 DMFAD用
    # # 步骤1：转置为 (m, n, n) → 让核数量作为第一维，方便批量展平
    # K_reshaped = yyK.transpose(2, 0, 1)  # 维度变化：(n, n, m) → (m, n, n)
    # # 步骤2：展平为 (m, n²) → 每个核矩阵变成1行（长度n×n）
    # K_flat = K_reshaped.reshape(P, -1)  # 维度变化：(m, n, n) → (m, n²)
    # # 步骤3：矩阵乘法 → (m, n²) @ (n², m) = (m, m)，正好是M矩阵（两两核的Frobenius内积）
    # M = K_flat @ K_flat.T  # 等价于 M[pq] = Tr(Kp @ Kq)
    
    for t in range(1, max_iter + 1):
        # --- forward ---
        Z = X_aug @ gat.T
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        eta = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        # --- 目标值 ---
        yykm = kernel_eta_vectorized(yyK, eta)
        obj1 = objectfunction(yykm, graph_w)
        # obj2 = lambda_reg * np.sum(np.einsum('np,pq,nq->n', eta, M, eta))

        # 增加正则化项
        # obj = obj1 + obj2
        obj = obj1

        obj_history.append(obj)
        
        # --- 早停判断 ---
        if obj < best_obj:
            best_obj = obj
            best_gat = gat.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and t > 10:
                break
        
        # --- backward (向量化版本) ---
        dEta = grad_objective_eta_vectorized(eta, yyK, graph_w)

        # # 增加正则化梯度
        # dEta += 2 * lambda_reg * (eta @ M.T)  # (N,P) @ (P,P) = (N,P)
        
        # 使用einsum进行高效计算
        J_terms = np.einsum('ni,nj->nij', eta, eta)  # s_i * s_j
        diag_eta = np.einsum('ni->ni', eta)  # 对角线元素
        eye = np.eye(P)[np.newaxis, :, :]  # 单位矩阵扩展
        J = eye * diag_eta[:, :, np.newaxis] - J_terms
        
        # 计算dZ
        dZ = np.einsum('np,npq->nq', dEta, J)
        
        # 计算梯度
        grad_gat = np.einsum('nq,nd->qd', dZ, X_aug)
        
        # --- Adam 更新 ---
        m = beta1 * m + (1 - beta1) * grad_gat
        v = beta2 * v + (1 - beta2) * (grad_gat ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        gat = gat - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    # --- 输出最优结果 ---
    gat = best_gat
    Z_final = X_aug @ gat.T
    Z_final = Z_final - np.max(Z_final, axis=1, keepdims=True)
    eta_opt = np.exp(Z_final) / np.sum(np.exp(Z_final), axis=1, keepdims=True)
    
    return gat, np.array(obj_history), eta_opt

def objective_value_eta_vectorized(eta, Km, W):
    """向量化版本的目标函数计算"""
    Keta = kernel_eta_vectorized(Km, eta)
    d = np.diag(Keta)
    objMat = d[:, None] + d[None, :] - 2 * Keta
    return np.sum(objMat * W)

def grad_objective_eta_vectorized(eta, Km, W):
    """向量化版本的梯度计算"""
    N, P = eta.shape
    dEta = np.zeros((N, P))
    W1 = np.sum(W, axis=1)
    
    # 使用向量化计算替代循环
    K_diag = np.diagonal(Km, axis1=0, axis2=1).T  # (P, N) -> (N, P)
    term1 = (K_diag * W1[:, None]) * eta
    
    # 使用einsum高效计算 term2
    # (K * W) @ eta 对于每个m
    term2 = np.einsum('ijm,ij,jm->im', Km, W, eta)
    
    dEta = 4 * term1 - 4 * term2
    return dEta

def kernel_eta_vectorized(Km, eta):
    """向量化版本的核组合"""
    # 使用einsum高效计算: sum_m (eta[:,m] * eta[:,m]^T * Km[:,:,m])
    return np.einsum('im,jm,ijm->ij', eta, eta, Km)