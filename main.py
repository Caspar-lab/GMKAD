import numpy as np
import scipy.io as sio
import os
import time
import csv  # 新增：导入csv模块
import argparse
from sklearn.model_selection import KFold
import warnings
import create_model
from test_model import test_model
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
import Granular_ball.GB as GB
warnings.filterwarnings('ignore')
# main 备份
# ------------------- 新增：CSV保存工具函数 -------------------
def save_alpha_auc_to_csv(file_path, result_dict):
    """
    保存alpha和对应的AUC结果到CSV（追加模式）
    :param file_path: CSV文件路径
    :param result_dict: 字典，包含列名和对应值
    """
    # 固定列名（数据集、噪声水平、alpha、最优AUC）
    # fieldnames = ['数据集名称', '噪声水平', 'alpha', '最优AUC']
    fieldnames = ['数据集名称', 'lam', '噪声水平', 'JNN', '最优AUC']
    # 判断文件是否存在，不存在则写表头
    file_exists = os.path.exists(file_path)
    
    with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # 首次写入表头
        writer.writerow(result_dict)

def min_max_normalize(data):
    """Min-max归一化函数"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_data

def crossvalind(method, labels, k):
    """模拟MATLAB的crossvalind函数"""
    if method == 'kfold':
        n_samples = len(labels)
        indices = np.zeros(n_samples, dtype=int)
        fold_size = n_samples // k
        for i in range(k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < k - 1 else n_samples
            indices[start_idx:end_idx] = i + 1
        return indices
    return None

def main(
        dataset_filter=None,
        fixed_jnn= None,
        fixed_sigma= None,
        fixed_degree= None,
        output_root='./exp_redo_W',
        dataset_root=r'D:\Microsoft\documents\博士课题\异常检测\实验\datasets',
        lambda_reg=0.0001,
        random_seed=42,
        matlab_compatible=False):
    """运行 GMKAD 参数实验。

    默认参数保持原来的全数据集/全网格行为。命令行传入固定参数时，可在
    独立输出目录中复现单个配置，而不会覆盖 ``exp_results_2``。
    """
    # 初始化变量
    res_all = []
    
    data_dict = sio.loadmat('new_all_datalists_outlier.mat')
    datalists = data_dict['datalists']
    no_data_ID = [8, 9, 19, 27, 33, 35, 39] + list(range(48, 50)) + list(range(63, 69))
    trandataset_path = dataset_root
    Alg_name = 'GMKAD'
 
    used_datasets = [
                    'cardio',

                    'cardiotocography_2and3_33_variant1',

                    'diabetes_tested_positive_26_variant1',

                    'ecoli',

                    'ionosphere_b_24_variant1',

                    'iris_Irisvirginica_11_variant1',

                    'musk',

                    'pageblocks_1_258_variant1',

                    'pendigits',

                    'pima_TRUE_55_variant1',

                    'satellite',

                    'sonar_M_10_variant1',

                    'waveform_0_100_variant1',

                    'wbc_malignant_39_variant1',

                    'wdbc_M_39_variant1',

                    'wine',

                    'wpbc_variant1',

                    'yeast_ERL_5_variant1',

                    'annealing_variant1',

                    'arrhythmia_variant1',

                    'bands_band_16_variant1',

                    'bands_band_27_variant1',

                    'bands_band_34_variant1',

                    'bands_band_42_variant1',

                    'hepatitis_2_9_variant1',

                    'nhanes_age_364',

                    'psdas_dropout_1421',

                    'thyroid_disease_variant1'                   
                     ]  # 可扩展其他数据集
    np.random.seed(random_seed)

    # ------------------- 新增：定义CSV保存路径 -------------------
    # csv_file_path = './alpha_auc_results.csv'
    # csv_file_path = './JNN_auc_results.csv'
    os.makedirs(output_root, exist_ok=True)
    csv_file_path = os.path.join(output_root, 'auc_results.csv')

    for i in range(len(datalists)):
        if i in no_data_ID:
            continue
        data_nameori = datalists[i][0][0]

        if data_nameori is None or data_nameori == '':
            continue

        if dataset_filter is not None and data_nameori != dataset_filter:
            continue
            
        dname = data_nameori
        if dname not in used_datasets:
            print(f'Dataset:{dname}不是目标数据集')
            continue
        
        folder_name = os.path.join(output_root, dname)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        elif any(os.scandir(folder_name)):
            print(f'Folder {folder_name} already exists. Skipping dataset {dname}.')
            continue
        data_path = os.path.join(trandataset_path, f'{data_nameori}.mat')
        
        if not os.path.exists(data_path):
            print(f'Dataset:{data_nameori} 在datasets目录中不存在！')
            continue
        
        # 加载数据
        try:
            data = sio.loadmat(data_path)
            tot_data = data['trandata']
        except:
            print(f'无法加载文件: {data_path}')
            continue
        
        data_only = tot_data[:, :-1]
        data_labels = tot_data[:, -1]
        data_label = data_labels.copy()
        
        # 数据预处理
        pos_class = 1
        target_mask = data_label == 0
        target_data = data_only[target_mask, :]
        target_data = np.column_stack([target_data, np.zeros(target_data.shape[0])])
        outlier_mask = data_label == 1
        outlier_data = data_only[outlier_mask, :]
        outlier_data = np.column_stack([outlier_data, np.ones(outlier_data.shape[0])])
        
        # 加噪处理
        X_orig = data_only
        y_labels = data_labels
        n_total = X_orig.shape[0]
        m_feat = X_orig.shape[1]
        noise_levels = np.arange(0.00, 0.1, 0.05)  # 噪声水平：0.0, 0.05, ..., 0.3
        rng_seed = 42
        cumulative = True
        num_levels = len(noise_levels)
        target_data_noisy = [None] * num_levels
        outlier_data_noisy = [None] * num_levels
        X_accumulate = X_orig.copy()
        
        for i_n in range(num_levels - 1):
            noise_level = round(noise_levels[i_n], 2)  # 当前噪声水平
            print(f'\n===== 数据集：{dname} | 噪声水平：{noise_level} =====')
            
            if not cumulative:
                X_cur = X_orig.copy()
            else:
                X_cur = X_accumulate.copy()
            
            if noise_level > 0:
                noise_n = int(np.ceil(noise_level * n_total))
                if noise_n > 0:
                    samples = np.random.choice(n_total, noise_n, replace=False)
                    random_m = np.random.randint(0, m_feat)
                    min_val = np.min(X_cur[samples, random_m])
                    max_val = np.max(X_cur[samples, random_m])
                    if max_val == min_val:
                        new_vals = min_val * np.ones(noise_n)
                    else:
                        new_vals = min_val + (max_val - min_val) * np.random.rand(noise_n)
                    X_cur[samples, random_m] = new_vals
            
            if cumulative:
                X_accumulate = X_cur.copy()
            
            # 重建目标数据和异常数据
            td = X_cur[y_labels != pos_class, :]
            td = np.column_stack([td, np.zeros(td.shape[0])])
            od = X_cur[y_labels == pos_class, :]
            od = np.column_stack([od, np.ones(od.shape[0])])
            target_data = target_data_noisy[i_n] = td
            outlier_data = outlier_data_noisy[i_n] = od
            
            tot_fold = 1
            m1, n1 = target_data.shape
            m2, n2 = outlier_data.shape
            ind_pos_all = crossvalind('kfold', target_data[:m1, n1-1], 1)
            ind_neg_all = crossvalind('kfold', outlier_data[:m2, n2-1], 1)
            
            # 初始化参数
            num_dis_knn = 19
            num_KNN = 1
            num_sigm = 7
            num_k = 4
            idx = 1
            opt_tmp = 0
            n_total_samples = data_only.shape[0]
            opt_scores = np.zeros((n_total_samples, 2))
            opt_auc = 0
            opt_Time = 0
            
            # ------------------- 新增：alpha参数循环（0-1，步长0.1） -------------------
            alphas = np.arange(0, 1, 2)  # 包含0.0到1.0，共11个值
            for alpha in alphas:
                alpha = round(alpha, 1)  # 避免浮点精度问题（如0.2999999999）
                print(f'--- 当前alpha：{alpha} ---')
                
                # 初始化该alpha下的最优AUC
                alpha_best_auc = 0.0
                
                # 原参数循环（JNN、run、fold、sigma、k）
                jnn_values = [fixed_jnn] if fixed_jnn is not None else np.arange(2, 60)
                for JNN in jnn_values:

                    # 初始化该alpha下的最优AUC
                    JNN_best_auc = 0.0

                    dis_knn = JNN
                    for run in [1]:
                        ind_pos = ind_pos_all
                        ind_neg = ind_neg_all
                        trauc = []
                        trtimes = []
                        
                        for fold in range(1, tot_fold + 1):
                            # 测试和训练索引
                            test_posind = (ind_pos == fold)
                            train_posind = test_posind
                            test_negind = (ind_neg == fold)
                            
                            # 选择数据
                            test_pos = target_data[test_posind, :]
                            test_neg = outlier_data[test_negind, :]
                            train_pos = target_data[train_posind, :]
                            
                            # 准备训练和测试数据
                            train_data = train_pos[:, :-1]
                            train_lbls = train_pos[:, -1]
                            test_data = np.vstack([test_pos[:, :-1], test_neg[:, :-1]])
                            test_lbls = np.hstack([test_pos[:, -1], test_neg[:, -1]])
                            train_data = test_data.copy()
                            train_lbls = np.ones(train_data.shape[0])
                            
                            # 归一化
                            train_data_m = min_max_normalize(train_data)
                            test_data_m = min_max_normalize(test_data)

                            if matlab_compatible:
                                split_threshold = round(np.sqrt(train_data_m.shape[0]))
                                centers, weights, index = GB.getGranularBall(
                                    train_data_m,
                                    min_split_num=split_threshold,
                                    secondary_split=False
                                )
                            else:
                                centers, weights, index = GB.getGranularBall(train_data_m)
                            
                            # 计算图权重（核心修改：将0.5改为alpha）
                            # num_data = train_data_m.shape[0]
                            num_data = centers.shape[0]
                            train_lbls_GB = np.ones(num_data)
                            graph_weights = np.zeros((num_data, num_data))
                            # # # 使用weights计算图权重
                            # for i in range(num_data):
                            #     for j in range(num_data):
                            #         graph_weights[i, j] = (weights[i] + weights[j]) / 2
                            # 归一化图权重到0-1范围
                            # max_weight = np.max(graph_weights)
                            # if max_weight > 0:
                            #     graph_weights = graph_weights / max_weight

                            # # 欧氏距离矩阵
                            # dists = squareform(pdist(train_data_m, metric='euclidean'))
                            # # 转换为相似度矩阵
                            # graph_weights = 1 - dists / train_data_m.shape[1]
                            # # 按alpha阈值过滤（关键修改）
                            # graph_weights[graph_weights < alpha] = 0
                            # # 其他位置设为1
                            # graph_weights[graph_weights >= alpha] = 1

                            graph_weights = np.ones((num_data, num_data))
                            
                            # 遍历sigma和k参数
                            full_sig_array = 2.0 ** np.arange(-3, 4)  # power(2, -3:3)
                            if fixed_sigma is None:
                                sigma_items = list(enumerate(full_sig_array))
                            else:
                                matching_sigma = np.flatnonzero(
                                    np.isclose(full_sig_array, fixed_sigma)
                                )
                                if matching_sigma.size != 1:
                                    raise ValueError(
                                        f'sigma={fixed_sigma} 不在 {full_sig_array.tolist()} 中'
                                    )
                                sigma_items = [(int(matching_sigma[0]), float(fixed_sigma))]
                            # sig_array = 2.0 ** np.arange(-3, -2)  # power(2, -3:3)
                            for sigm_idx, sigm_val in sigma_items:
                                degree_values = [fixed_degree] if fixed_degree is not None else range(1, 5)
                                for k in degree_values:  # k=1~4
                                # for k in range(1, 2):  # k=1~4
                                    # for reg_lambda in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 100]:
                                    for reg_lambda in [lambda_reg]:
                                        start_time = time.time()
                                        
                                        # 参数设置
                                        final = [f'g{sigm_val}', f'p{k}']
                                        
                                        # 训练模型
                                        model = create_model.create_model(
                                            centers,
                                            train_lbls_GB,
                                            final,
                                            graph_weights,
                                            # reg_lambda
                                        )
                                        if matlab_compatible:
                                            model['kernel_scale_divisor'] = 3.0
                                        outlier_scores = test_model(test_data_m, model, JNN, centers, index, weights)
                                        time_tmp = time.time() - start_time
                                        
                                        # 计算AUC
                                        auc_temp = roc_auc_score(test_lbls, outlier_scores)
                                        trauc.append(auc_temp)
                                        trtimes.append(time_tmp)

                                        save_tmp = np.ones((test_data_m.shape[0], 2))
                                        save_tmp[:, 0] = outlier_scores
                                        save_tmp[0, 1] = auc_temp
                                        save_tmp[1, 1] = time_tmp

                                        results_name1 = f'{data_nameori}_{Alg_name}_JNN{JNN}_sigm{sigm_val}_k{k}.mat'
                                        results_path1 = os.path.join(folder_name, results_name1)
                                        sio.savemat(results_path1, {'out_scores': save_tmp})

                                        
                                        # 输出当前参数组合的结果
                                        print(f'[JNN={JNN}, sigma={sigm_val}, k={k}, lam={reg_lambda} AUC={auc_temp:.4f}, time={time_tmp:.2f}]')
                                        
                                        # 更新该alpha下的最优AUC
                                        if auc_temp > JNN_best_auc:
                                            JNN_best_auc = auc_temp
                                        
                                        # 更新全局最优（原逻辑保留）
                                        if auc_temp > opt_auc:
                                            opt_scores[:, 0] = outlier_scores
                                            opt_auc = auc_temp
                                            opt_Time = time_tmp
                                            opt_scores[0, 1] = opt_auc
                                            opt_scores[1, 1] = opt_Time
                                            opt_scores[2, 1] = sigm_idx + 1
                                            opt_scores[3, 1] = JNN
                                            opt_scores[4, 1] = k
                                        
                                        # 清理变量
                                        del model, outlier_scores
                        
                        # 该alpha下所有参数组合完成，输出最优AUC
                        print(f'JNN={JNN} 最优AUC：{JNN_best_auc:.4f}')
                        
                        # ------------------- 新增：保存当前alpha的结果到CSV -------------------
                        result_dict = {
                            '数据集名称': dname,
                            'lam': lambda_reg,
                            '噪声水平': noise_level,
                            # 'alpha': alpha,
                            'JNN': JNN,
                            '最优AUC': round(JNN_best_auc, 4)
                        }
                        save_alpha_auc_to_csv(csv_file_path, result_dict)
                        
                        trauc = np.array(trauc)
                        trtimes = np.array(trtimes)
                        max_auc = np.max(trauc)
                        best_auc = max_auc * 100
                        best_time = trtimes[np.argmax(trauc)]
                        opt_tmp = max(opt_tmp, best_auc)
                        print(f'{data_nameori} 噪声{noise_level} 最优auc为：{opt_tmp:.4f}')
                        
                        del trauc, trtimes
                
                # 保存最优结果到mat（原逻辑保留）
                # results_name2 = f'noise//noise_{i_n+1}_{data_nameori}_{Alg_name}.mat'
                results_name2 = f'{data_nameori}_{Alg_name}.mat'
                results_path2 = os.path.join(folder_name, results_name2)
                sio.savemat(results_path2, {'opt_scores': opt_scores})

def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='运行 GMKAD 实验；可固定为单数据集、单参数配置。'
    )
    parser.add_argument('--dataset', dest='dataset_filter')
    parser.add_argument('--jnn', dest='fixed_jnn', type=int, default=2)
    parser.add_argument('--sigma', dest='fixed_sigma', type=float)
    parser.add_argument('--degree', dest='fixed_degree', type=int)
    parser.add_argument('--output-root', default='./exp_redo2')
    parser.add_argument(
        '--dataset-root',
        default=r'C:\OD\Shihao\datasets'
    )
    parser.add_argument('--lambda-reg', type=float, default=0.0001)
    parser.add_argument('--seed', dest='random_seed', type=int, default=42)
    parser.add_argument(
        '--matlab-compatible',
        action='store_true',
        help='使用参考 MATLAB 的粒球阈值、关闭二次分裂，并将融合核除以 3。'
    )
    return parser


if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
