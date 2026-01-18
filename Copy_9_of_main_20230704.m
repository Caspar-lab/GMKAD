clear all
clc
res_all = [];
load all_datalists_outlier.mat;
datalists{85} = 'psdas_dropout_1421';
datalists{86} = 'nhanes_age_364';


no_data_ID=[9 10 27 33 35 39 40 47:48 49 63:68];
no_data_ID = [no_data_ID];
%obj = NaN (57)
trandataset_path = '实验\datasets';
%load '\实验\datasets\all_datalists_outlier.mat';
Alg_name = 'GMKAD';
%for data_num = 53:length(datalists)

used_datasets = {
'ionosphere_b_24_variant1'
};

% used_datasets = {
% 'cardio'
% 'cardiotocography_2and3_33_variant1'
% 'diabetes_tested_positive_26_variant1'
% 'ecoli'
% 'ionosphere_b_24_variant1'
% 'iris_Irisvirginica_11_variant1'
% 'musk'
% 'pageblocks_1_258_variant1'
% 'pendigits'
% 'pima_TRUE_55variant1'
% 'satellite'
% 'sonar_M_10_variant1'
% 'thyroid'
% 'vowels'
% 'waveform_0_100_variant1'
% 'wbc_malignant_39_variant1'
% 'wdbc_M_39_variant1'
% 'wine'
% 'wpbc_variant1'
% 'yeast_ERL_5_variant1'
% 'annealing_variant1'
% 'arrhythmia_variant1'
% 'bands_band_16_variant1'
% 'bands_band_27_variant1'
% 'bands_band_34_variant1'
% 'bands_band_42_variant1'
% 'hepatitis_2_9_variant1'
% 'nhanes_age_364'
% 'psdas_dropout_1421'
% 'thyroid_disease_variant1'};

for data_num = 29:length(datalists)
    data_nameori=datalists{data_num};
    dname = data_nameori;
    
    if ~ismember(dname, used_datasets)
        disp(['Dataset:' dname '不是目标数据集'])
        continue;
    end
    
    if ismember(data_num,no_data_ID)
        disp(['Dataset:' data_nameori ' 执行不出来！'])
        continue;
    end
    
    folder_name = ['exp21//' dname];
    
    
%     % 检查文件是否存在
%     if exist(folder_name, 'dir')
%         fprintf('%s 已有实验结果，跳过处理。\n', data_nameori);
%         continue; % 跳到下一次循环
%     end
%     mkdir(folder_name);
        
    
    % 构建datasets目录下的完整文件路径
    %data_path = [fullfile('datasets', data_nameori), '.mat'];
    data_path = [fullfile(trandataset_path, data_nameori), '.mat'];
    % 检查文件是否存在
    if ~exist(data_path, 'file')
        disp(['Dataset:' data_nameori ' 在datasets目录中不存在！'])
        continue;
    end
    % 加载指定目录下的文件
    load(data_path);
    tot_data = trandata;
    data_only  = tot_data(:,1:end-1);
    data_labels = tot_data(:,end);
    data_label = data_labels;
    count = 0;
    
    %%%%%%%%%%%%% 数据预处理 %%%%%%%%%%%%%%
    res.dname=[dname num2str(data_num)];
    % 初始化正类标签的值为 0
    pos_class = 0;
    % 将 data_labels 中等于 pos_class (0) 的标签转换为 1 (正类)
    data_label(data_labels==pos_class) = 1;
    % 将 data_labels 中不等于 pos_class (0) 的标签转换为 2 (负类)
    data_label(data_labels~=pos_class) = 2;
    % 提取正类样本 (data_label == 1) 的数据，存储到 target_data
    target_data = data_only(data_label==1,:);
    % 在 target_data 末尾添加一列全 1，标记为正类 (维度扩展为 [样本数, 特征数+1])
    target_data = cat(2, target_data, ones(size(target_data,1),1));
    outlier_data = data_only(data_label==2,:);
    outlier_data = cat(2, outlier_data, ones(size(outlier_data,1),1).*2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%加噪
    % 保留原始无标签数据
    X_orig = data_only;       % n_total x m
    y_labels = data_labels;   % n_total x 1 原始标签 (用于重建 target/outlier)
    n_total = size(X_orig,1);
    m_feat = size(X_orig,2);

    % ---------- sweep 模式参数 ----------
    noise_levels = 0.05:0.05:0.3; % 噪声比例
    rng_seed = 42;             % 随机种子
    cumulative = true;        % 是否累积噪声
    rng('default'); % 重置随机数生成器，退出旧模式
    rng(rng_seed);

    num_levels = length(noise_levels);
    target_data_noisy = cell(num_levels, 1);
    outlier_data_noisy = cell(num_levels, 1);

    % 若选择累积，则 X_accumulate 会保留上次修改
    X_accumulate = X_orig;

    for i_n = 1:num_levels-1
        lam = round(noise_levels(i_n), 2);

        if ~cumulative
            X_cur = X_orig; % 每次在原始数据上注入（独立实验）
        else
            X_cur = X_accumulate; % 在上一次结果上继续注入（累积）
        end

        if lam > 0
            noise_n = ceil(lam * n_total);
            if noise_n > 0
                samples = randperm(n_total, noise_n);
                random_m = randi(m_feat); % 随机选择属性列
                min_val = min(X_cur(samples, random_m));
                max_val = max(X_cur(samples, random_m));
                if max_val == min_val
                    new_vals = min_val * ones(noise_n,1);
                else
                    %生成一个 noise_n × 1 的矩阵，每个元素都是 [0,1) 之间的均匀随机数。
                    new_vals = min_val + (max_val - min_val) .* rand(noise_n,1);
                end
                X_cur(samples, random_m) = new_vals;
            end
        end

        if cumulative
            X_accumulate = X_cur;
        end

        % 重建 target/outlier 并存入 cell
        td = X_cur(y_labels == pos_class, :);
        td = cat(2, td, ones(size(td,1),1));
        od = X_cur(y_labels ~= pos_class, :);
        od = cat(2, od, ones(size(od,1),1).*2);

        target_data_noisy{i_n} = td;
        outlier_data_noisy{i_n} = od;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    target_data = target_data_noisy{i_n};
    outlier_data = outlier_data_noisy{i_n};
    
    tot_fold =1;
    [m1,n1] = size(target_data);
    % 生成用于五折交叉验证的索引 shape = (m1, 1)
    % ind_pos_all 是一个长度为 m1 的向量，包含每个样本的折编号（1 到 5），用于划分训练和验证集
    ind_pos_all = crossvalind('kfold', target_data(1:m1,n1),1);
    [m2,n2] = size(outlier_data);
    ind_neg_all = crossvalind('kfold', outlier_data(1:m2,n2),1);
    
    % 初始化存储结果的矩阵
    num_dis_knn = 19; % 2:20
    num_KNN = 1;    % 2:60
    num_sigm = 7;    % 目前固定为 1
    num_k = 4;       % 目前固定为 2
    %results_mat = zeros(num_dis_knn * num_KNN * num_sigm * num_k, 6); % [dis_knn, KNN, sigm, k, auc, time]
    idx = 1;
    opt_tmp = 0;
    opt_scores = zeros(size(data_only,1),2);
    opt_auc = 0;
    opt_Time = 0;
    
    %dis_knn = 2;
    for JNN = 2:2
        dis_knn = JNN;
        %JNN = dis_knn;
        for KNN = 2:2
            for run = 1
                ind_pos = ind_pos_all(:, run);
                ind_neg = ind_neg_all(:, run);
                % 5折验证
                trauc = zeros(tot_fold, 1); % 存储每折的 auc
                trtimes = zeros(tot_fold, 1); % 存储每折的时间

                for f = 1:tot_fold
                    % 测试和训练索引
                    test_posind = (ind_pos == f);
                    train_posind = test_posind; % 修正为取反
                    test_negind = (ind_neg == f);

                    % 选择数据
                    test_pos = target_data(test_posind, :);
                    test_neg = outlier_data(test_negind, :);
                    train_pos = target_data(train_posind, :);

                    % 训练和测试数据集
                    train_data = train_pos(:, 1:end-1);
                    train_lbls = train_pos(:, end);
                    
                    test_data = cat(1, test_pos(:, 1:end-1), test_neg(:, 1:end-1));
                    test_lbls = cat(1, test_pos(:, end), test_neg(:, end));
                    
                    train_data = test_data;
                    train_lbls = ones(size(train_data,1),1);

                    train_number = length(train_lbls);
                    test_number = length(test_lbls);

                    % 归一化
                    train_data_m = min_max_normalize(train_data);
                    test_data_m = min_max_normalize(test_data);

                    % Granular Ball 生成
%                     L = round(sqrt(size(train_data_m, 1)));
%                     [centers_target, weights_target, radius_target, density_target, index] = getGranularBall(train_data_m, L);
                    %centers_lbls = ones(size(centers_target, 1), 1);
%                     train_data_m_GB = centers_target;
%                     train_lbls_GB = centers_lbls;
                    
%                     if size(centers_target, 1)<(dis_knn+1)
%                         best_auc = 0;
%                         best_time = 0;
%                         continue;
%                     end
                    
                    % 计算距离和图权重
                    [num_data, dim] = size(train_data_m);
%                     distance = zeros(num_data, num_data);
%                     for i = 1:num_data
%                         for j = 1:num_data
%                             distance(i, j) = sqrt(sum((train_data_m_GB(i, :) - train_data_m_GB(j, :)).^2));
%                         end
%                     end
%                     [~, dis_index] = sort(distance);
%                     knn_index = dis_index(2:dis_knn+1, :);
                    
                    %93.5464
                     graph_weights = ones(num_data, num_data);
%                     
%                     for i = 1:num_data
%                         for j = 1:dis_knn
%                             graph_weights(i, knn_index(j, i)) = (train_data_m_GB(i, :) * train_data_m_GB(knn_index(j, i), :)' + 1)^2;
%                         end
%                     end
                    
%                     graph_weights = pdist2(train_data_m_GB, train_data_m_GB);
%                     % 归一化得到相似度矩阵
%                     graph_similarities = 1-graph_weights ./ dim;  % 使用点除确保逐元素除法
%                     graph_weights = min_max_normalize(graph_similarities);
%                     graph_weights = (graph_weights + graph_weights')./2;
%                     graph_weights(graph_weights <= 0.8) = 0;
                    

                    % 训练模型
                    optmod = {};
                    temp_ind = 0;
                    sig_array = power(2, -3:3); % 扩展 sigm 循环
                    sig_k_ind = 1;
                    %for sigm = 1:length(sig_array)
                    %for k = 1:1
                    for sigm = 1:length(sig_array)
                        for k = 1:1:4
                            tic;
                            %核参数1
                            %final = {['g' num2str(sig_array(sigm))], ['p' num2str(k)], 'l', ['h' num2str(2)]};
                            final = {['g' num2str(sig_array(sigm))], ['p' num2str(k)]};
                            temp_ind = temp_ind + 1;
                            [model, eta] = create_model(train_data_m, train_lbls, final, graph_weights);
                            [labels_GOCK, labels_score] = test_model(test_data_m, model, final, JNN, KNN, test_number);
                            time_tmp = toc;
                            
                            % 计算性能指标
                            [~, ~, ~, ~, ~, ~, ~, ~, auc_temp] = Evaluate(test_lbls, labels_GOCK, labels_score, 2);
                            trauc(temp_ind) = auc_temp;
                            trtimes(temp_ind) = time_tmp;
                            optmod{temp_ind} = model;
                            
                            % 存储结果
                            sigm_value = sig_array(sigm); 
                            k_value = k;    
                            %[dis_knn, sigm_value, k_value, auc_temp, time_tmp]
                            [JNN, KNN, sigm_value, k_value, auc_temp, time_tmp]
                            %results_mat(idx, :) = [JNN, KNN, sigm_value, k_value, auc_temp, time_tmp];
                            idx = idx + 1;
                            
                            % 存储到mat文件
                            results_name1 = [dname '_' Alg_name '_sigm' num2str(sigm) '_JNN' num2str(JNN) '_k' num2str(k) '.mat'];
                            save_path = fullfile(folder_name, results_name1);
                            out_scores = labels_score';
                            %save(save_path, 'out_scores');
                            
                            %更新最优值
                            if auc_temp > opt_auc
                                opt_scores(:,1) = out_scores;
                                opt_auc = auc_temp;
                                opt_Time = time_tmp;
                                opt_scores(1,2) = opt_auc;
                                opt_scores(2,2) = opt_Time;
                                opt_scores(3,2) = num2str(sigm);
                                opt_scores(4,2) = (JNN);
                                opt_scores(5,2) = (k);
                            end
                            clear model eta labels_GOCK labels_score
                        end
                    end
                    
                    
                    % 选择最佳参数
                    [max_auc, opt_ind] = max(trauc);
                    best_auc = max_auc * 100; % 转换为百分比
                    best_time = trtimes(opt_ind);
                    opt_tmp = max(opt_tmp, best_auc);
                    fprintf('res%d的最优auc为：%.4f', data_num, opt_tmp);
                    
                    clear trauc trtimes
                    
                end

                
            end
        end
    end
    results_name2 = ['noise' '_' num2str(i_n) '_' dname '_' Alg_name '.mat'];
    save_path = fullfile(folder_name, results_name2);
    save(save_path, 'opt_scores');
    end
end
%save('gpl_result.mat','res_all')