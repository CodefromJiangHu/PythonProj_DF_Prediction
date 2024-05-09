# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from DataPostPrecessingControl import post_date_processing_lstm_for_predic, get_acc_with_lstm
import DataPreDrocessingControl as dpdCL
import ExcelControl as excelCL

def get_data_from_excel(df_excel, row_size,colum_size,list_fields_names,list_label,flag_traindata):
    # 属性表字段名
    # list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_rootdep", "f_soildep", "f_soiltrength", "f_vegload", "Gully_J_Sd",
    #               "流速Sd", "流深Sd"]
    # list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_soiltrength", "f_vegload", "Gully_J_Sd", "流速Sd", "流深Sd"]
    # list_label = ["label"]
    # 初始化全流域矩阵
    X_total = np.zeros(shape=[row_size, colum_size], dtype='float32')
    Y_total = np.zeros(shape=[row_size, 1], dtype="bool")
    X_total = excelCL.convertDFtoArray(df_excel, list_fields_names)
    # print(X_yangben)
    if flag_traindata:
        Y_total = excelCL.convertDFtoArray(df_excel, list_label)
    return X_total, Y_total
# 数组左侧填充默认值
def pad_default_values(original_list,wanted_length,padded_value):
    # original_list = [1, 2, 3, 4]
    new_array = np.pad(original_list, (wanted_length - len(original_list), 0), 'constant', constant_values=(padded_value,))
    print(new_array)
    return new_array

# str序列转化为lstm类型的张量矩阵，并填充默认值
def convert_listStr_into_tensor(list_str,time_steps,padded_value):
    tensor = []
    for idx in range(len(list_str)):
        str_feature = list_str[idx]
        list_feature = str(str_feature).split(",")
        # 转换为float型
        list_float_array = [float(item) for item in list_feature]
        if len(list_float_array)<= time_steps:
            list_feature_with_time_steps = pad_default_values(list_float_array, time_steps, padded_value)
        else:
            list_feature_with_time_steps = list_float_array[-time_steps:]
        tensor.append(np.transpose(list_feature_with_time_steps))
    return np.array(tensor)

# 随机打乱lstm的动态数组部分
def shuffle_dynamic_feature_value(X_dynamic):
    tem_X_dynamic = X_dynamic.reshape(-1,1).tolist()
    np.random.shuffle(tem_X_dynamic)
    X_dynamic_shuffled = tem_X_dynamic
    return np.array(X_dynamic_shuffled).reshape(X_dynamic.shape)
# 随机打乱n次动态特征数组的元素，求平均模型精度
def get_mean_acc_by_shuffle_dynamic_features(model, X_test_dynamic, X_test_static, Y_test_static, n_cycle):
    list_acc = []
    for idx in range(n_cycle):
        X_test_dynamic_shuffled = shuffle_dynamic_feature_value(X_test_dynamic)
        tem_acc = get_acc_with_lstm(model, [X_test_dynamic_shuffled, X_test_static], Y_test_static)
        list_acc.append(tem_acc)
    return np.mean(list_acc)
# 随机打乱静态特征数组的第j列元素
def shuffle_static_feature_values(X_static, idx_cols):
    new_X_static = X_static
    cols_values = X_static[:, idx_cols]
    np.random.shuffle(cols_values)
    cols_values_shuffled = cols_values
    new_X_static[:, idx_cols] = cols_values_shuffled
    return new_X_static
# 随机打乱n次静态特征数组的元素，求平均模型精度
def get_mean_acc_by_shuffle_static_features(model, X_test_dynamic, X_test_static, Y_test_static, idx_cols, n_cycle):
    list_acc = []
    for idx in range(n_cycle):
        X_test_static_shuffled = shuffle_static_feature_values(X_test_static, idx_cols)
        tem_acc = get_acc_with_lstm(model, [X_test_dynamic, X_test_static_shuffled], Y_test_static)
        list_acc.append(tem_acc)
    return np.mean(list_acc)
# 输出各个指标对模型精度的贡献度
def output_weights_of_indicators(list_mean_acc_shuffled, list_indicatiors_name, acc_standard, opt_filePath):
    list_acc_dif_abs = np.abs(acc_standard - np.array(list_mean_acc_shuffled))
    tem_sum = np.sum(list_acc_dif_abs)
    importance_rate = list_acc_dif_abs/tem_sum
    new_DF = pd.DataFrame(np.array(importance_rate).reshape(1,-1), columns=list_indicatiors_name)
    new_DF.to_excel(opt_filePath, index=False)

if __name__ == '__main__':
    # controlling flag
    flag_prediction = True
    flagSimle = False
    flag_fit_scaler = False

    # parameters detial info
    str_storm_name = "2022Storm_062006"
    modelNmae = str_storm_name+"_rc24_GroupC_AVOA_DL-DF-LSTM"
    df_totalyangben_environment = pd.read_excel('input/Storm_DynamicAnalysis/step5p0_total_' + str_storm_name + '_RC_case.xlsx')

    # parameters detial info
    row_size = df_totalyangben_environment.shape[0]
    colum_size = 14

    # setting runtime name
    str_para_vary = "run_optimal_dl-df-lstm_"

    # reading rainfall list
    time_steps = 24
    list_rainfall_str_total = df_totalyangben_environment["Str_R"].tolist()
    dynamic_feature_total = np.reshape(
        convert_listStr_into_tensor(list_rainfall_str_total, time_steps=time_steps, padded_value=0),
        (row_size, time_steps, 1))

    # reading samples
    list_fields_names = ["CJ", "RH", "AI",
                         "AP", "AT", "LT", "DF", "SE", "SD",
                         "NDVI", "LD", "RD", "PD", "EP72"]
    list_fields_names_plot = ["CG", "RH", "AI",
                         "AP", "AT", "LT", "DF", "SE", "SD",
                         "NDVI", "LD", "RD", "PD", "EP72"]
    list_label = ["Target"]
    static_X_total, static_Y_total = get_data_from_excel(df_totalyangben_environment, row_size, colum_size, list_fields_names,
                                           list_label, False)

    # 数字编码标签转为独热编码 The digital coding label is converted to a unique thermal coding
    static_Y_total = tf.keras.utils.to_categorical(static_Y_total)

    # (1)samples preprocessing
    saved_scaler_filepath = "saved_para/saved_scaler_for_ipv_lstm_1.0.pkl"
    if flag_fit_scaler:
        df_totalyangben_environment_forStandard = pd.read_excel('output/step5p0_total_RC_total.xlsx')
        static_X_total_for_sd, static_Y_total_for_sd = get_data_from_excel(df_totalyangben_environment_forStandard, row_size, colum_size,
                                                             list_fields_names,
                                                             list_label, False)
        sc_first = dpdCL.dataprocessing_first_getStandardScaler(static_X_total_for_sd)
        dpdCL.save_scaler(sc_first, saved_scaler_filepath)

    sc = dpdCL.load_scaler(saved_scaler_filepath)
    static_X_total_std = dpdCL.dataprocessing_second_StandardScaler_transform(sc, static_X_total)

    if flag_prediction:
        # obtaining trainted model
        model_saved_path = 'saved_model/model_ds_lstm_rc24_v0.2_final_10.h5'
        # model loading
        trained_model = tf.keras.models.load_model(model_saved_path)
        model_best = trained_model
        # prediction and validation
        post_date_processing_lstm_for_predic(model_best, modelNmae, [dynamic_feature_total, static_X_total_std], str_para_vary,
                                  flagSimle, df_totalyangben_environment)

