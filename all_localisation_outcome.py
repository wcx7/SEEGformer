import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, precision_recall_curve, auc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from mne.io import concatenate_raws, read_raw_edf
from utils import fft
import patient_information
import ChannelBasedTransformer_fft_PE
import ChannelBasedTransformer_fft
import json
from sklearn.metrics import ndcg_score
import random
import re


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.targets[index]
        return sample, target


def precision_at_k(y_true, k):
    # 计算前k正确率
    return np.sum(y_true[0:k])/k

def recall_at_k(y_true, k):
    # 计算召回率
    return np.sum(y_true[0:k]) / np.sum(y_true)

def f1_at_k(y_true, k):
    prec = precision_at_k(y_true, k)
    rec = recall_at_k(y_true, k)
    # 计算F1分数
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def averaged_precision_at_k(y_true, k):
    total = 0
    for i in range(k):
        total += precision_at_k(y_true, i+1)
    return total/k

def loc_evaluation(patient, result_dict, fold, include_2):
    " fold: 'fold1','fold2'...,'all' "
    # 选择mid或ave
    if scoring_method == 'mid':
        sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))

    # ground_truth的位置
    if include_2 == 1:
        ground_truth = patient.EZ + patient.EZ2
    else:
        ground_truth = patient.EZ
    # 找到键的位置
    positions = {key: index + 1 for index, key in enumerate(sorted_dict) if key in ground_truth}
    # 打印结果
    y_scores = []
    for key, value in sorted_dict.items():
        y_scores += [value]
    y_true = [0]*len(ch_names)
    print("Positions of keys:")
    for key, position in positions.items():
        y_true[position-1] = 1
        print(f"{key}: {position}")
    print('total rank:', sum(positions.values()))
    print(f'total channel number:{len(ch_names)}')

    # 计算指标
    # 计算 AUPRC
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(y_true, y_scores)
    # 打印结果
    print(f"AUPRC: {auprc}")
    print(f"AUROC: {auroc}")
    all_localisation_dict[f'{patient_name}'][fold] = {}
    all_localisation_dict[f'{patient_name}'][fold]['AUPRC']= auprc
    all_localisation_dict[f'{patient_name}'][fold]['AUROC']= auroc
    # 计算k的值
    k_values = [round(len(ch_names)*0.02), round(len(ch_names)*0.05), round(len(ch_names)*0.1)]
    j_values = ['2%', '5%', '10%']
    # 使用 zip 函数同时迭代 k_values 和 j_values
    for k, j in zip(k_values, j_values):
        precision_k = precision_at_k(y_true, k)
        a_precision_k = averaged_precision_at_k(y_true, k)
        recall_k = recall_at_k(y_true, k)
        f1_k = f1_at_k(y_true, k)
        all_localisation_dict[f'{patient_name}'][fold][f"Precision@{j}"]= precision_k
        all_localisation_dict[f'{patient_name}'][fold][f"Averaged Precision@{j}"]= a_precision_k
        all_localisation_dict[f'{patient_name}'][fold][f"Recall@{j}"]= recall_k
        all_localisation_dict[f'{patient_name}'][fold][f"F1@{j}"]= f1_k
        print(f"Precision@{j}: {precision_k}")
        print(f"Averaged Precision@{j}: {a_precision_k}")
        print(f"Recall@{j}: {recall_k}")
        print(f"F1@{j}: {f1_k}")


    if hasattr(patient, 'EZ3'):
        # 计算k的值
        len_ch_names = len(ch_names)  # 节点总数
        len_EZ = len(patient.EZ)      # 相关性为0.6的节点数
        len_EZ2 = len(patient.EZ2)    # 相关性为0.3的节点数
        len_EZ3 = len(patient.EZ3)    # 相关性为0.1的节点数
        # 创建相关性分数列表
        true_relevances = [0.6] * len_EZ + [0.3] * len_EZ2 + [0.1] * len_EZ3
        true_relevances += [0.0] * (len_ch_names - len(true_relevances))  # 填充剩余的节点为0
        # 转换为numpy数组以便使用sklearn
        true_relevances = np.array(true_relevances)
        predicted_scores = []
        for i in range(len_EZ):
            predicted_scores += [sorted_dict[patient.EZ[i]]]
        for i in range(len_EZ2):
            predicted_scores += [sorted_dict[patient.EZ2[i]]]
        for i in range(len_EZ3):
            predicted_scores += [sorted_dict[patient.EZ3[i]]]
        # 创建一个集合，包含所有需要排除的键
        excluded_keys = set(patient.EZ + patient.EZ2 + patient.EZ3)
        # 通过列表推导，选取不在 excluded_keys 集合中的字典键对应的值
        other_list = [value for key, value in sorted_dict.items() if key not in excluded_keys]
        predicted_scores += other_list
        # 计算NDCG
        ndcg_value = ndcg_score([true_relevances], [predicted_scores])
        all_localisation_dict[f'{patient_name}'][fold][f"NDCG"] = ndcg_value
        print("NDCG Value:", ndcg_value)
    else:
        # 计算k的值
        len_ch_names = len(ch_names)  # 节点总数
        len_EZ = len(patient.EZ)      # 相关性为0.65的节点数
        len_EZ2 = len(patient.EZ2)    # 相关性为0.35的节点数
        # 创建相关性分数列表
        true_relevances = [0.65] * len_EZ + [0.35] * len_EZ2
        true_relevances += [0.0] * (len_ch_names - len(true_relevances))  # 填充剩余的节点为0
        # 转换为numpy数组以便使用sklearn
        true_relevances = np.array(true_relevances)
        predicted_scores = []
        for i in range(len_EZ):
            predicted_scores += [sorted_dict[patient.EZ[i]]]
        for i in range(len_EZ2):
            predicted_scores += [sorted_dict[patient.EZ2[i]]]
        # 创建一个集合，包含所有需要排除的键
        excluded_keys = set(patient.EZ + patient.EZ2)
        # 通过列表推导，选取不在 excluded_keys 集合中的字典键对应的值
        other_list = [value for key, value in sorted_dict.items() if key not in excluded_keys]
        predicted_scores += other_list
        # 计算NDCG
        ndcg_value = ndcg_score([true_relevances], [predicted_scores])
        all_localisation_dict[f'{patient_name}'][fold][f"NDCG"] = ndcg_value
        print("NDCG Value:", ndcg_value)
    
    # 计算随机NDCG
    if fold == 'all':
        values_list = list(sorted_dict.values())
        ndcg_value = []
        for i in range(10000):
            random.shuffle(values_list)
            random_scores = np.array(values_list)
            ndcg_value += [ndcg_score([true_relevances], [random_scores])]
        all_localisation_dict[f'{patient_name}'][fold][f"Random NDCG"] = sum(ndcg_value)/10000
        print("Random NDCG Value:", sum(ndcg_value)/10000)
        print()

# 打分方式  'mid' or 'ave'
scoring_method = 'mid'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

patient_name_list = [ ]
data_agu_list = [ ]
all_localisation_dict = {}
save_path = ' '
for i in range(len(patient_name_list)):
    patient_name = patient_name_list[i]
    print(patient_name)
    all_localisation_dict[f'{patient_name}'] = {}
    # 获得通道名称ch_names
    patient = getattr(patient_information, patient_name)()
    if patient.PE == 1:
        position_encoding = np.load(f'../position_encoding_{patient_name}.npy')
    filename = next(iter(patient.seizure_start_dict.keys()))
    useless_chan = patient_information.useless_chan
    exclude = patient.exclude
    exclude = useless_chan + exclude

    ch_names = patient.ch_names
    for p in range(len(ch_names)):
        # 移除 'EEG '
        ch_names[p] = ch_names[p].replace('EEG ', '')

        # 如果字符串不以 "POL " 开头，则添加 "POL "
        if not ch_names[p].startswith('POL '):
            ch_names[p] = 'POL ' + ch_names[p]
        # 使用正则表达式移除 '-' 之后的所有字符
        ch_names[p] = re.sub('-.*', '', ch_names[p])

   
    # 数据准备 发作   ***评价时推荐使用数据增强之前的原始数据，即窗宽1s，步长1s***
    dataset_path = f'../preprocessed_data_1s_1s(Z-Score)/'
    data = np.load(os.path.join(dataset_path,'seizure/all_data.npy'))
    data = np.transpose(data,(2, 0, 1))
    labels = np.load(os.path.join(dataset_path,'seizure/label.npy'))

    batch_size = 32
    n_chans = data.shape[1]
    num_heads = 4
    embed_dim = 64   # 64
    mlp_ratio = 2
    dropout = 0
    num_blocks = 2
    num_classes = 2
    fs = 1000
    fft_points = data.shape[2]*2
    freq_used = 100  # 只使用 freq_used Hz以下的信息
    fft_dim = int(freq_used*fft_points/fs)


    # 定位效果定量评价
    # 主文件夹路径
    main_folder_path = f'../{patient_name}'
    # 子文件夹名称
    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    dataset_dict = {'dataset1':CustomDataset(data, labels),'dataset2':CustomDataset(data, labels),
                    'dataset3':CustomDataset(data, labels),'dataset4':CustomDataset(data, labels),'dataset5':CustomDataset(data, labels)}
    frequency = np.zeros([5, len(ch_names)])
    average_score = np.zeros([5, len(ch_names)])
    i=0

    # 使用 for 循环依次访问这五个子文件夹
    for fold in folds:
        i += 1
        # 构造当前 fold 的完整路径
        current_fold_path = os.path.join(main_folder_path, fold)
        if patient.PE == 1:
            model = ChannelBasedTransformer_fft_PE.ChannelBasedTransformer_fft(num_heads, fft_dim, embed_dim, mlp_ratio, dropout, n_chans, num_blocks, num_classes, position_encoding, pool = 'cls')
        else:
            model = ChannelBasedTransformer_fft.ChannelBasedTransformer_fft(num_heads, fft_dim, embed_dim, mlp_ratio, dropout, n_chans, num_blocks, num_classes, pool = 'cls')
        model.load_state_dict(torch.load(current_fold_path + '/train_fft.pth'))
        model.eval()        # 将模型切换到评估模式
        model.to(device)
        dataset = dataset_dict[f'dataset{i}']
        data_loader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle=False, drop_last = False)
        # 初始化空列表以保存预测和真实标签
        all_predictions = []
        all_targets = []
        all_attention_scores_real_testset = []
        all_attention_scores_imag_testset = []
        all_attention_scores_abs_testset = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                _, x_real, x_imag, x_abs = fft(inputs, fs, fft_points, freq_used)
                x_real = torch.Tensor(x_real).to(device)
                x_imag = torch.Tensor(x_imag).to(device)
                x_abs = torch.Tensor(x_abs).to(device)
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上
                outputs,attn_weights_real, attn_weights_imag, attn_weights_abs = model(x_real, x_imag, x_abs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.tolist())
                all_targets.extend(labels.tolist())
                all_attention_scores_real_testset.append(attn_weights_real)
                all_attention_scores_imag_testset.append(attn_weights_imag)
                all_attention_scores_abs_testset.append(attn_weights_abs)

        # 测试集上的模型表现
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')
        
        combined_tensor = torch.cat(all_attention_scores_real_testset, dim=0) + torch.cat(all_attention_scores_imag_testset, dim=0)
        combined_tensor = combined_tensor.cpu().numpy()
        combined_tensor_mean = np.mean(combined_tensor, axis=1)
        attention_score = combined_tensor_mean
        min_val = np.min(attention_score)
        max_val = np.max(attention_score)
        attention_score = attention_score.T
        # Min-Max标准化
        attention_score = attention_score[1:,:]   # 去除cls
        scaled_attention_score = (attention_score - min_val) / (max_val - min_val)
        if scoring_method == 'mid':
            #中位数法
            q1 = np.percentile(scaled_attention_score, 50)
            frequency_temp = np.sum(scaled_attention_score > q1, axis=1)
            frequency_temp = (frequency_temp - np.min(frequency_temp)) / (np.max(frequency_temp) - np.min(frequency_temp))
            if patient_name == 'WangZezhi':
                frequency_temp = frequency_temp[ :-2]
            frequency[i-1,:] = frequency_temp
            result_dict_mid = {}
            for k in range(len(ch_names)):
                result_dict_mid[ch_names[k]] = frequency_temp[k]
            if scoring_method == 'mid':
                loc_evaluation(patient, result_dict_mid, fold, include_2)
        else:
            #平均法
            channel_total = np.mean(scaled_attention_score, axis = 1)
            channel_total = (channel_total - np.min(channel_total)) / (np.max(channel_total) - np.min(channel_total))
            average_score[i-1,:] = channel_total
            result_dict_ave = {}
            for k in range(len(ch_names)):
                result_dict_ave[ch_names[k]] = channel_total[k]
            if scoring_method == 'ave':
                loc_evaluation(patient, result_dict_ave, fold, include_2)
    if scoring_method == 'mid':
        #中位数法
        stds = np.std(frequency, axis=0)
        frequency = np.mean(frequency, axis=0)
        result_dict_mid = {}
        for k in range(len(ch_names)):
            result_dict_mid[ch_names[k]] = frequency[k]
        loc_evaluation(patient, result_dict_mid, 'all', include_2)
    else:
        #平均法
        stds = np.std(average_score, axis=0)
        average_score = np.mean(average_score, axis=0)
        result_dict_ave = {}
        for k in range(len(ch_names)):
            result_dict_ave[ch_names[k]] = average_score[k]
        loc_evaluation(patient, result_dict_ave, 'all', include_2)

    # 将字典保存为json格式，并使得其格式易读
    if scoring_method == 'mid':
        with open(f'{save_path}/localisation_eval_mid.json', 'w') as json_file:
            json.dump(all_localisation_dict, json_file, indent=4)
    else:
        with open(f'{save_path}/localisation_eval_ave.json', 'w') as json_file:
            json.dump(all_localisation_dict, json_file, indent=4)    
  