import pandas as pd
import numpy as np
from mne.io import concatenate_raws, read_raw_edf
import os
import torch
import pywt
import sys
import patient_information 

def count_files_in_directory(folder_path):
    if not os.path.exists(folder_path):
        return 0  # 目录不存在，没有文件
    if not os.path.isdir(folder_path):
        return 0  # 指定的路径不是目录
    file_count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_count += 1  # 如果项目是文件，增加文件计数
        elif os.path.isdir(item_path):
            # 如果项目是目录，可以选择递归计算该目录中的文件数
             file_count += count_files_in_directory(item_path)
    return file_count

# 切数据
def crop_data(original_path,low_freq,high_freq,seizure_list,end_list,window_len,step, resampled_freq, patient_name, save_path, exclude = None ):
    """ 
    original_path: path where the original .edf file is.
    start,end: start and end moment. unit:s
    low_freq,high_freq: band pass filter.
    seizure_list,end_list: a list of onset/offset moment. unit:s
    window_len: window length of the eeg, to descripe save path. int (unit: s)
    step: Step size of Seizure sliding window. unit: s.
    resampled_freq: 降采样频率
    exclude: 排除掉哪些通道
    """
    window_len = int(window_len*resampled_freq)
    step = int(step*resampled_freq)
    raw = read_raw_edf(original_path,preload=False,encoding='latin1',exclude=exclude)


    for k in range(len(seizure_list)):
        if 2*seizure_list[k]-end_list[k] > 0:
            raw1 = raw.copy().crop((2*seizure_list[k]-end_list[k]),end_list[k]).load_data()   # 如想保留raw，用raw.copy().crop(start[0],end[0]).load_data()， 截取和发作时间等长的发作前数据
            # 应用带通滤波,应当先滤波，再降采样
            raw1.filter(low_freq, high_freq)
            # 应用陷波滤波
            raw1 = raw1.notch_filter([50,100], picks='all', filter_length='auto',
                                                phase='zero', verbose=True)
            # 降采样到1000Hz
            raw1 = raw1.resample(sfreq=resampled_freq)    
            data = raw1.get_data(units='uV')
            # 对整个裁出来的原数据矩阵进行Z-score标准化（用未发作的数据做标准化）
            seizure_len = int(end_list[k]*resampled_freq - seizure_list[k]*resampled_freq)   
            mean = np.mean(data[:, :(data.shape[1]-seizure_len)])
            std = np.std(data[:, :(data.shape[1]-seizure_len)])
            data = (data - mean) / std
            del raw1

            # 切数据preseizure
            folder_path = os.path.join(save_path,patient_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            folder_path = os.path.join(save_path,f'{patient_name}/preseizure_{int(window_len/resampled_freq)}s_{step/resampled_freq}s/')
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            i = count_files_in_directory(folder_path)    # 已有文件的数量
            n = 0

            while(n*(step)<data.shape[1]-seizure_len-window_len):
                win_start = n*(step) 
                win_end = win_start + window_len
                temp = data[:,win_start:win_end]
                filename = os.path.join(save_path,f'{patient_name}/preseizure_{int(window_len/resampled_freq)}s_{step/resampled_freq}s/window_{i}.npy')
                np.save(filename, temp)
                i += 1
                n += 1

            # 切数据seizure
            folder_path = os.path.join(save_path,f'{patient_name}/seizure_{int(window_len/resampled_freq)}s_{step/resampled_freq}s/')
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            i = count_files_in_directory(folder_path)    # 已有文件的数量
            n = 0
            while(n*step<seizure_len-window_len):
                win_start = data.shape[1]-seizure_len + n*step
                win_end = win_start + window_len
                temp = data[:,win_start:win_end]
                filename = os.path.join(save_path,f'{patient_name}/seizure_{int(window_len/resampled_freq)}s_{step/resampled_freq}s/window_{i}.npy')
                np.save(filename, temp)
                i += 1
                n += 1
    return()            

# 数据整合、打标签
def integer(read_path,method,data_class,window_len,step,patient_name):
    """ 
    read_path: path where the croped data is.
    method: method of nomlization, 'Z-Score' or 'Min-Max' or 'None'.
    data_class: if preseizure, data_class = 0 ; if seizure, data_class = 1. int
    win_len: window length of the eeg, used to descripe save path. int (unit: second)
    """
    label = []
    file_names = os.listdir(read_path)
    file_len = len(file_names)
    temp = np.load(os.path.join(read_path, file_names[0]))
    data = np.zeros([temp.shape[0], temp.shape[1], file_len])
    for i in range(len(file_names)):
        data[:,:,i] = np.load(os.path.join(read_path, file_names[i]))
        if data_class == 0:
            eeg_class = 'preseizure' 
            label += [0]
        if data_class == 1:
            eeg_class = 'seizure' 
            label += [1]
        print(f'{i+1}/{file_len}')
    folder_path1 = f'../{patient_name}/preprocessed_data_{window_len}s_{step}s('+method+')'
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)    
    folder_path2 = f'../{patient_name}/preprocessed_data_{window_len}s_{step}s('+method+')/'+eeg_class
    if not os.path.exists(folder_path2):
        os.mkdir(folder_path2)
    np.save(folder_path2+'/all_data.npy',data)
    np.save(folder_path2+'/label.npy',label)
    return ()


if __name__ == '__main__':
    patient_name_list = [ ]
    data_agu_list = [ ]
    save_path = ' '
    data_agu_list = [1]
    for i in range(len(patient_name_list)):
        patient_name = patient_name_list[i]   # 患者姓名 
        patient = getattr(patient_information, patient_name)()
        useless_chan = patient_information.useless_chan
        exclude = patient.exclude
        exclude = useless_chan + exclude

        low_freq = 0.5
        high_freq = 100
        window_len = 1                # 窗长
        step = data_agu_list[i]       # 窗移动的步长
        resampled_freq = 1000         # 重采样的frequency

        for key,value in patient.seizure_start_dict.items():
            original_path = f' '
            seizure_list = value      # 这段脑电中发作癫痫开始时刻的列表
            end_list = patient.seizure_end_dict[key]         # 这段脑电中发作癫痫结束时刻的列表
            crop_data(original_path,low_freq,high_freq,seizure_list,end_list,window_len,step, resampled_freq, patient_name, save_path, exclude = exclude)

        # 整合
        read_path = f' '   # 切好的clips的路径，步长和clip长度要改
        method = 'Z-Score'
        label = 0
        integer(read_path,method,label,window_len,step,patient_name)
        read_path = f' '    # 1.0  0.5 etc
        method = 'Z-Score'
        label = 1
        integer(read_path,method,label,window_len,step,patient_name)

