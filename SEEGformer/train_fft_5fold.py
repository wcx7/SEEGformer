import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import patient_information
import ChannelBasedTransformer_fft_PE 
import ChannelBasedTransformer_fft
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import os
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import datetime
from utils import fft
import json
 


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



patient_name_list = [ ]
data_agu_list = [1]*len(patient_name_list)
epoch_list = [30]*len(patient_name_list)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for k in range(len(patient_name_list)):
    # 数据准备
    patient_name = patient_name_list[k]
    patient = getattr(patient_information, patient_name)()
    dataset_path = f'../{patient_name}/preprocessed_data_1s_{data_agu_list[k]}s(Z-Score)'
    data = np.concatenate((np.load(os.path.join(dataset_path,'preseizure/all_data.npy')),np.load(os.path.join(dataset_path,'seizure/all_data.npy'))), axis=2)
    data = np.transpose(data,(2, 0, 1))
    data_labels = np.concatenate((np.load(os.path.join(dataset_path,'preseizure/label.npy')),np.load(os.path.join(dataset_path,'seizure/label.npy'))))
    position_encoding = np.load(f'../position_encoding_{patient_name}.npy')

    # 初始化 KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 使用 KFold 划分数据
    for fold, (train_index, test_index) in enumerate(kf.split(data), 1):
        save_path1 = f'../{patient_name}'
        if not os.path.exists(save_path1):
                os.mkdir(save_path1)
        save_path = f'../{patient_name}/fold{fold}'
        if not os.path.exists(save_path):
                os.mkdir(save_path)
        train_data, test_data = data[train_index,:], data[test_index,:]
        train_labels, test_labels = data_labels[train_index], data_labels[test_index]

        train_dataset = CustomDataset(train_data, train_labels)
        test_dataset = CustomDataset(test_data, test_labels)

        batch_size = 32
        learning_rate = 0.001  # 0.001
        learning_rate1 = learning_rate
        decay = 0.95
        epochs = epoch_list[k]
        lambda_l2 = 0.001  # l2正则化参数

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last = False)


        n_chans = data.shape[1]
        num_heads = 4
        embed_dim = 64   # 64
        mlp_ratio = 2
        dropout = 0.2
        num_blocks = 2
        num_classes = 2
        fs = 1000
        fft_points = data.shape[2]*2
        freq_used = 100  # 只使用 freq_used Hz以下的信息
        fft_dim = int(freq_used*fft_points/fs)
        patient.PE = 0
        if patient.PE == 1:
            model = ChannelBasedTransformer_fft_PE.ChannelBasedTransformer_fft(num_heads, fft_dim, embed_dim, mlp_ratio, dropout, n_chans, num_blocks, num_classes, position_encoding, pool = 'mean')
        else:
            model = ChannelBasedTransformer_fft.ChannelBasedTransformer_fft(num_heads, fft_dim, embed_dim, mlp_ratio, dropout, n_chans, num_blocks, num_classes, pool = 'mean')
        model.to(device)  
        criterion = nn.CrossEntropyLoss()
        train_accuracy_list = []
        test_accuracy_list = []

        # 当前时间
        formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        for epoch in range(epochs):
            print(patient_name_list[k])
            model.train()
            total_loss = 0
            optimizer = optim.Adam(model.parameters(), lr=learning_rate1)
            learning_rate1 = learning_rate1*decay
            # 初始化空列表以保存预测和真实标签
            all_predictions = []
            all_targets = []
            all_attention_scores_real_trainset = []
            all_attention_scores_imag_trainset = []
            all_attention_scores_abs_trainset = []

            for batch in train_loader:
                inputs, labels = batch
                _, x_real, x_imag, x_abs = fft(inputs,fs,fft_points,freq_used)
                x_real = torch.Tensor(x_real).to(device)
                x_imag = torch.Tensor(x_imag).to(device)
                x_abs = torch.Tensor(x_abs).to(device)
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上
                optimizer.zero_grad()
                outputs,attn_weights_real, attn_weights_imag, attn_weights_abs = model(x_real, x_imag, x_abs)   # 控制使用哪些部分
                loss = criterion(outputs, labels)
                # loss += ChannelBasedTransformer.l2_regularization(model, lambda_l2, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # 记录预测和真实标签
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.tolist())
                all_targets.extend(labels.tolist())
                if epoch == epochs-1:
                    all_attention_scores_real_trainset.append(attn_weights_real)
                    all_attention_scores_imag_trainset.append(attn_weights_imag)
                    all_attention_scores_abs_trainset.append(attn_weights_abs)

            # 训练损失与训练集上的模型表现
            avg_train_loss = total_loss / len(train_loader)
            accuracy = accuracy_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions, average='binary')
            recall = recall_score(all_targets, all_predictions, average='binary')
            f1 = f1_score(all_targets, all_predictions, average='binary')
            train_accuracy_list += [accuracy]
            print(f"Epoch [{epoch + 1}/{epochs}]")
            print('Train:')
            print(f"Accuracy:{accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Train Loss: {avg_train_loss:.4f}")


            # 在测试集上评估模型性能
            model.eval()
            # 初始化空列表以保存预测和真实标签
            all_logits = []
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for batch in test_loader:
                    inputs, labels = batch
                    _, x_real, x_imag, x_abs = fft(inputs,fs,fft_points,freq_used)
                    x_real = torch.Tensor(x_real).to(device)
                    x_imag = torch.Tensor(x_imag).to(device)
                    x_abs = torch.Tensor(x_abs).to(device)
                    inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上
                    outputs,attn_weights_real, attn_weights_imag, attn_weights_abs = model(x_real, x_imag, x_abs)  # 控制使用哪些部分
                    _, predicted = torch.max(outputs, 1)
                    logits = outputs.cpu().numpy()[:, 1]
                    all_logits.extend(logits.tolist())
                    all_predictions.extend(predicted.tolist())
                    all_targets.extend(labels.tolist())


            # 计算所有可能的阈值下的真正率（TPR）和假正率（FPR）
            fpr, tpr, thresholds = roc_curve(all_targets, all_logits)
            # 计算 AUROC 值
            auroc = auc(fpr, tpr)
            # 计算 AUPRC
            precision, recall, thresholds = precision_recall_curve(all_targets, all_logits)
            auprc = auc(recall, precision)

            # 测试集上的模型表现
            accuracy = accuracy_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions, average='binary')
            recall = recall_score(all_targets, all_predictions, average='binary')
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
            # 计算特异度
            specificity = tn / (tn + fp)
            f1 = f1_score(all_targets, all_predictions, average='binary')
            test_accuracy_list += [accuracy]

            print('Test:')
            print(f"Accuracy:{accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auroc:.4f}")
            print()


            if epoch == epochs-1:
                np.save(save_path + f'/all_targets.npy', all_targets)
                np.save(save_path + f'/all_logits.npy', all_logits)
                n_samples = test_data.shape[0]
                n_resamples = 200 
                all_auroc = np.zeros([n_resamples])
                all_auprc = np.zeros([n_resamples])
                all_acc = np.zeros([n_resamples])
                all_pre = np.zeros([n_resamples])
                all_rec = np.zeros([n_resamples])
                all_spe = np.zeros([n_resamples])
                all_f1 = np.zeros([n_resamples])
            
                for n in range(n_resamples):
                    # 随机生成索引，允许重复
                    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
                    # 根据索引重采样数据和标签
                    resampled_data = test_data[indices, :]
                    resampled_label = test_labels[indices]
                    test_dataset = CustomDataset(resampled_data, resampled_label)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last = False)
                    # 初始化空列表以保存预测和真实标签
                    all_logits = []
                    all_predictions = []
                    all_targets = []

                    with torch.no_grad():
                        for batch in test_loader:
                            inputs, labels = batch
                            _, x_real, x_imag, x_abs = fft(inputs,fs,fft_points,freq_used)
                            x_real = torch.Tensor(x_real).to(device)
                            x_imag = torch.Tensor(x_imag).to(device)
                            x_abs = torch.Tensor(x_abs).to(device)
                            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上
                            outputs,attn_weights_real, attn_weights_imag, attn_weights_abs = model(x_real, x_imag, x_abs)  # 控制使用哪些部分
                            _, predicted = torch.max(outputs, 1)
                            logits = outputs.cpu().numpy()[:, 1]
                            all_logits.extend(logits.tolist())
                            all_predictions.extend(predicted.tolist())
                            all_targets.extend(labels.tolist())

                    # 计算所有可能的阈值下的真正率（TPR）和假正率（FPR）
                    fpr, tpr, thresholds = roc_curve(all_targets, all_logits)
                    # 计算 AUROC 值
                    auroc = auc(fpr, tpr)
                    # 计算 AUPRC
                    precision, recall, thresholds = precision_recall_curve(all_targets, all_logits)
                    auprc = auc(recall, precision)
                    accuracy = accuracy_score(all_targets, all_predictions)
                    precision = precision_score(all_targets, all_predictions, average='binary')
                    recall = recall_score(all_targets, all_predictions, average='binary')
                    # 计算混淆矩阵
                    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
                    # 计算特异度
                    specificity = tn / (tn + fp)
                    f1 = f1_score(all_targets, all_predictions, average='binary')

                    all_auroc[n] = auroc
                    all_auprc[n] = auprc
                    all_acc[n] = accuracy
                    all_pre[n] = precision
                    all_rec[n] = recall
                    all_spe[n] = specificity
                    all_f1[n] = f1
                
                #保存
                result_dict_bootstrap = {"accuracy":all_acc.tolist(), "precision":all_pre.tolist(), 'recall':all_rec.tolist(), 'specificity':all_spe.tolist(),
                                "f1":all_f1.tolist(), "auroc":all_auroc.tolist(), "auprc":all_auprc.tolist()}
                # 保存各项指标文件
                with open(save_path + f'/result_dict_bootstrap.json', 'w') as json_file:
                    json.dump(result_dict_bootstrap, json_file)

        result_dict = {"accuracy":accuracy, "precision":precision, 'recall':recall, 'specificity':specificity,
                        "f1":f1, "auroc":auroc, "auprc":auprc}


        


        x = np.linspace(1,len(train_accuracy_list),num=len(train_accuracy_list))
        plt.figure(dpi=300)
        # 绘制training
        plt.plot(x, train_accuracy_list, label='train', color='blue', linestyle='-', marker='o', markersize=3)

        # 绘制testing
        plt.plot(x, test_accuracy_list, label='test', color='orange', linestyle='-', marker='o', markersize=3)

        # 添加标题和标签
        plt.title("Training process")
        plt.xlabel(f"Epoch\nbatch_size:{batch_size} learning_rate:{learning_rate} epochs:{epochs} num_heads:{num_heads}\n embed_dim:{embed_dim} mlp_ratio:{mlp_ratio} dropout:{dropout} num_blocks:{num_blocks}")
        plt.ylabel("Accuracy")
        plt.ylim(0.5,1)
        # 添加图例
        plt.legend()
        plt.tight_layout()

        # 保存模型
        formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 保存各项指标文件
        with open(save_path + f'/result_dict.json', 'w') as json_file:
            json.dump(result_dict, json_file)
        torch.save(model.state_dict(), save_path + f'/train_fft.pth')
        # 保存图表
        plt.savefig(save_path + f'/Training process')
