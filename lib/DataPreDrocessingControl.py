# 常用的包
import numpy as np
import pandas as pd
# 数据前处理需要的包
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
# 数据归一化和标准化
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
# 参数保存
import pickle


def minmax_matrix(data):
    #实例化一个转换器类
    transfer = MinMaxScaler(feature_range=[0,1])
    #调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)
    return data_new

def standard_matrix(data):
    #实例化一个转换器类
    transfer = StandardScaler()
    #调用fit_transform
    data_new = transfer.fit_transform(data)
    return data_new

def dataprocessing_simple(X_yangben, Y_taget, rate_split):
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    # smo = SMOTE(random_state=42)
    # X_smo, y_smo = smo.fit_resample(X_yangben, Y_taget)

    # 特征归一化
    sc = StandardScaler()
    sc.fit(X_yangben)

    #样本全集跟随变化
    X_yangben_std = sc.transform(X_yangben)

    # 2.拆分测试集、训练集。
    X_train, X_test, Y_train, Y_test = train_test_split(X_yangben_std, Y_taget, test_size=rate_split, random_state=0)
    # 设置随机数种子，以便比较结果。
    return X_train, X_test, Y_train, Y_test
# saving scaler
def save_scaler(ipt_scaler, saved_scaler_path_name):

    scaler_name = "input/scaler.pkl"
    scaler_name = saved_scaler_path_name
    # save scaler
    with open(scaler_name, 'wb') as f:
        pickle.dump(ipt_scaler, f)

# loading scaler
def load_scaler(loading_scaler_path_name):
    scaler_name = "input/scaler.pkl"
    scaler_name = loading_scaler_path_name
    # load scaler
    with open(scaler_name, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def dataprocessing_first_getStandardScaler(X_yangben):
    # 特征归一化
    sc = StandardScaler()
    sc.fit(X_yangben)
    # 设置随机数种子，以便比较结果。
    return sc
def dataprocessing_second_StandardScaler_transform(_StandardScaler,X_yangben):
    # 样本全集跟随变化
    X_yangben_std = _StandardScaler.transform(X_yangben)
    # 设置随机数种子，以便比较结果。
    return X_yangben_std
def dataprocessing_third_split(X_yangben_std, Y_taget, _rate_split, _random_state):
    # 2.拆分测试集、训练集。
    X_train, X_test, Y_train, Y_test = train_test_split(X_yangben_std, Y_taget, test_size=_rate_split, random_state=_random_state)
    # 设置随机数种子，以便比较结果。
    return X_train, X_test, Y_train, Y_test
# 二维数值样本的常规数据前处理，包括过SMOTE过采样、归一化、非线性处理、数据集划分
# X_yangben 二维矩阵，numpy类型
# Y_taget 二维矩阵，列数为1，numpy类型
# rate_split 样本随机划分后用于验证的比例
# 返回两套数据，一套用于模型训练，另外一套为数据未通过过采样处理的前处理数据
def dataprocessing_common(X_yangben, Y_taget, rate_split):
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(X_yangben, Y_taget)

    # 特征归一化
    sc = StandardScaler()
    sc.fit(X_smo)
    X_train_std = sc.transform(X_smo)

    #样本全集跟随变化
    X_yangben_std = sc.transform(X_yangben)

    #非线性处理
    poly_reg = PolynomialFeatures(degree=5)
    x_poly = poly_reg.fit_transform(X_train_std)
    #样本全集跟随变化
    X_yangben_poly = poly_reg.fit_transform(X_yangben_std)

    # 2.拆分测试集、训练集。
    X_train, X_test, Y_train, Y_test = train_test_split(x_poly, y_smo, test_size=rate_split, random_state=0)
    # 设置随机数种子，以便比较结果。
    return X_train, X_test, Y_train, Y_test, X_yangben_poly, Y_taget

def dataprocessing_LSTM(X_yangben, Y_taget, rate_split):
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(X_yangben, Y_taget)

    # 特征归一化
    sc = StandardScaler()
    sc.fit(X_smo)
    X_train_std = sc.transform(X_smo)

    #样本全集跟随变化
    X_yangben_std = sc.transform(X_yangben)

    # #非线性处理
    # poly_reg = PolynomialFeatures(degree=5)
    # x_poly = poly_reg.fit_transform(X_train_std)
    # #样本全集跟随变化
    # X_yangben_poly = poly_reg.fit_transform(X_yangben_std)

    # 2.拆分测试集、训练集。
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_std, y_smo, test_size=rate_split, random_state=0)
    # 设置随机数种子，以便比较结果。
    return X_train, X_test, Y_train, Y_test, X_yangben_std, Y_taget

# 流域范围内提取众数，并剔除nodata值
def statistics_major(masked_featureMap,nodata_value,num_multiply):
    # 格式处理
    # 0 删除nodata值
    masked_featureMap_one = masked_featureMap[np.logical_not(masked_featureMap == nodata_value)]
    # 1 数组压扁
    # masked_featureMap = masked_featureMap.reshape([-1, 0])
    # 整型化
    masked_featureMap_two = masked_featureMap_one*num_multiply
    masked_featureMap_three = masked_featureMap_two.astype(int)
    # 2 求众数
    # major = stats.mode(masked_featureMap.)

    # bincounts = np.bincount(masked_featureMap)
    # 3 返回结果
    # major = np.argmax(bincounts)
    major = -1
    if len(masked_featureMap)>0:
        major = stats.mode(masked_featureMap_three)[0][0]
    else:
        major = 0
    major = float(major/num_multiply)
    return major
# 流域内含一定数量的栅格值为key值，则返回key值，否则返回流域内众数
def statistics_major_with_key(masked_featureMap,nodata_value,num_multiply,key_value,min_numbers_of_key):
    major = -1
    # 格式处理
    # 0 删除nodata值
    masked_featureMap_one = masked_featureMap[np.logical_not(masked_featureMap == nodata_value)]
    # 获取与key值一致的栅格值list
    list_valuse_eql_key = masked_featureMap_one[(masked_featureMap_one == key_value)]
    # list长度大于最小限制数量时，则众数赋值key值，主要用于解决流域内含一定量的建筑物，但是建筑物的面积往往不是占面积比例最高的土地利用类型
    if len(list_valuse_eql_key) >= min_numbers_of_key:
        major = key_value
    else:
        # 1 数组压扁
        # masked_featureMap = masked_featureMap.reshape([-1, 0])
        # 整型化
        masked_featureMap_two = masked_featureMap_one*num_multiply
        masked_featureMap_three = masked_featureMap_two.astype(int)
        # 2 求众数
        # major = stats.mode(masked_featureMap.)

        # bincounts = np.bincount(masked_featureMap)
        # 3 返回结果
        # major = np.argmax(bincounts)

        if len(masked_featureMap)>0:
            major = stats.mode(masked_featureMap_three)[0][0]
        else:
            major = 0
        major = float(major/num_multiply)
    return major
# 流域范围内提取均值，并剔除nodata值
def statistics_mean(masked_featureMap,nodata_value):
    # 0 删除nodata值
    masked_featureMap = masked_featureMap[np.logical_not(masked_featureMap == nodata_value)]
    # 1 数组压扁
    # masked_featureMap = masked_featureMap.reshape([-1, 0])
    # 2 求平均
    if len(masked_featureMap)>0:
        tem_mean = np.mean(masked_featureMap)
    else:
        tem_mean = 0.0
        # 3 返回结果
    return tem_mean
# 取map_within_flag为1的部分进行统计
def statistics_dif_max_min_within_flag(masked_featureMap,nodata_value,map_within_flag):
    # 0 删除nodata值
    masked_featureMap = masked_featureMap*map_within_flag
    masked_featureMap = masked_featureMap[np.logical_not(masked_featureMap == nodata_value)]
    # 1 数组压扁
    # masked_featureMap = masked_featureMap.reshape([-1, 0])
    # 2 求和
    # tem_mean = np.mean(masked_featureMap)
    if len(masked_featureMap)>0:
        tem_max = np.max(masked_featureMap)
        tem_min = np.min(masked_featureMap)
    else:
        tem_max = 0.0
        tem_min = 0.0
    # 3 返回结果
    return tem_max - tem_min

# 流域范围内提出最大值减去最小值
def statistics_dif_max_min(masked_featureMap,nodata_value):
    # 0 删除nodata值
    masked_featureMap = masked_featureMap[np.logical_not(masked_featureMap == nodata_value)]
    # 1 数组压扁
    # masked_featureMap = masked_featureMap.reshape([-1, 0])
    # 2 求和
    # tem_mean = np.mean(masked_featureMap)
    if len(masked_featureMap) > 0:
        tem_max = np.max(masked_featureMap)
        tem_min = np.min(masked_featureMap)
    else:
        tem_max = 0.0
        tem_min = 0.0
    # 3 返回结果
    return tem_max-tem_min

def statistics_gullymouth_mean(masked_featureMap, masked_flowAccu, scale_half_size, nodata_value):
    # 1、获取汇流累积量的最大值
    maximum = np.max(masked_flowAccu)
    # 2、获取沟口位置的行列号
    index = np.where(masked_flowAccu == maximum)
    index = np.array(index).reshape([-1])
    # 3、获取沟口周围的栅格矩阵
    # 获取输入矩阵的行列数
    m, n = masked_featureMap.shape
    sub_m_region, top_m_region = 0, 0
    sub_n_region, top_n_region = 0, 0
    # 行号范围
    if (index[0]-scale_half_size >= 0) and (index[0]+scale_half_size <= m-1):
        sub_m_region = index[0]-scale_half_size
        if sub_m_region<0:
            sub_m_region = 0
        top_m_region = index[0]+scale_half_size+1
    elif index[0]-scale_half_size < 0:
        sub_m_region = 0
        top_m_region = index[0]+scale_half_size*2+1
    else:
        sub_m_region = index[0]-scale_half_size*2
        if sub_m_region<0:
            sub_m_region = 0
        top_m_region = m+1
    #列号范围
    if (index[1]-scale_half_size >= 0) and (index[1]+scale_half_size <= n-1):
        sub_n_region = index[1]-scale_half_size
        if sub_n_region<0:
            sub_n_region = 0
        top_n_region = index[1]+scale_half_size+1
    elif index[1]-scale_half_size < 0:
        sub_n_region = 0
        top_n_region = index[1]+scale_half_size*2+1
    else:
        sub_n_region = index[1]-scale_half_size*2
        if sub_n_region<0:
            sub_n_region = 0
        top_n_region = n+1

    gullymouth_region_featuremap = masked_featureMap[sub_m_region:top_m_region, sub_n_region:top_n_region]
    # print(gullymouth_region_featuremap)
    # 剔除nodata
    gullymouth_region_featuremap = gullymouth_region_featuremap[np.logical_not(gullymouth_region_featuremap == nodata_value)]
    if not(len(gullymouth_region_featuremap) > 0):
        tem_mean = 0.000001
    else:
        # 获取平均值
        tem_mean = np.mean(gullymouth_region_featuremap)
    return tem_mean

# 随机过采样，并输出过采样结果
def do_RandomOverSampler(old_X, old_Y,list_fieldNames,excelFilePath):
    from imblearn.over_sampling import RandomOverSampler
    from ExcelControl import construct_DF
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(old_X, old_Y)
    y_resampled = np.array(y_resampled).reshape(-1,1)
    matrix_X = np.concatenate([X_resampled, y_resampled], axis=1)

    # 过采样结果输出
    df = construct_DF(list_fieldNames, matrix_X)
    df.to_excel(excelFilePath)
    print("随机过采样的结果已保存为excel:"+excelFilePath)
    return X_resampled, y_resampled

# 随机过采样，并输出过采样结果
def do_RandomUnderSampler(old_X, old_Y,list_fieldNames,excelFilePath):
    from imblearn.under_sampling import RandomUnderSampler
    from ExcelControl import construct_DF
    ros = RandomUnderSampler(sampling_strategy=1)
    X_resampled, y_resampled = ros.fit_resample(old_X, old_Y)
    y_resampled = np.array(y_resampled).reshape(-1,1)
    matrix_X = np.concatenate([X_resampled, y_resampled], axis=1)

    # 次采样结果输出
    df = construct_DF(list_fieldNames, matrix_X)
    df.to_excel(excelFilePath,index=False)
    print("随机过采样的结果已保存为excel:"+excelFilePath)
    return X_resampled, y_resampled

# 随机过采样，并输出过采样结果
def do_SMOTE(old_X, old_Y,list_fieldNames,excelFilePath):
    from imblearn.over_sampling import SMOTE
    from ExcelControl import construct_DF
    ros = SMOTE(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(old_X, old_Y)
    y_resampled = np.array(y_resampled).reshape(-1,1)
    matrix_X = np.concatenate([X_resampled, y_resampled], axis=1)

    # 过采样结果输出
    df = construct_DF(list_fieldNames, matrix_X)
    df.to_excel(excelFilePath, index=False)
    print("随机过采样的结果已保存为excel:"+excelFilePath)
    return X_resampled, y_resampled

# 降雨数据前处理
# 将逐小时的降雨序列转为n小时的累积降雨序列
def convert_hrs_into_nhrs(list_pre_hrs, para_n):
    """
    list_pre_hrs:逐小时的降雨序列
    para_n :n小时，需要累积的小时间隔
    """
    list_nhr_pr = np.zeros(int(len(list_pre_hrs) / para_n))
    for index in range(para_n):
        list_nhr_pr = list_nhr_pr + list_pre_hrs[index:len(list_pre_hrs):para_n]
    return list_nhr_pr



if __name__ == '__main__':
    flag_statistics_major = False
    flag_RandomOverSampler = False
    flag_SMOTE = False
    flag_RandomUnderSampler = True

    if flag_statistics_major:
        tem_matrix = [[1.0,4.0,1.0,1.0,1.0],
                      [1,2,8,5,1],
                      [1,3,7,4,1],
                      [1,3,4,6,1],
                      [1.0,3,1,1,1.0]]
        acc_matrix = [[1,4,1,1,1],
                      [1,2,8,5,1],
                      [100,3,1,4,1],
                      [1,3,4,6,1],
                      [1,3,1,1,1]]
        tem_matrix = np.array(tem_matrix)
        # tem_matrix = tem_matrix*10
        # tem_matrix = tem_matrix.astype(int)
        # print(tem_matrix[0:2,:])
        # mean = statistics_gullymouth_mean(tem_matrix, acc_matrix, 1, 1)
        # print(mean)
        nodata = 1

        major = statistics_major(tem_matrix,nodata,10000)
        print(major)

    if flag_RandomOverSampler:
        fieldsName = ["VOLUME", "TTGP", "PFV", "Vba", "Distance_to_Glacier", "Glacier Volume", "Slope",
                      "probability_To_Channel", "IC_Mean", "Pre", "Cul_tem"]
        list_label = ["Target"]
        list_fieldNames = ["VOLUME", "TTGP", "PFV", "Vba", "Distance_to_Glacier", "Glacier Volume", "Slope",
                      "probability_To_Channel", "IC_Mean", "Pre", "Cul_tem","Target"]
        df_input = pd.read_excel("input/Yangben_GlacialDF.xlsx")
        filePath = "Yangben_GlacialDF_OS.xlsx"
        # 获取矩阵
        import ExcelControl as EXLctl
        matrix_X = EXLctl.convertDFtoArray(df_input, fieldsName)
        matrix_Y = EXLctl.convertDFtoArray(df_input, list_label)
        do_RandomOverSampler(matrix_X, matrix_Y, list_fieldNames, filePath)

    if flag_SMOTE:
        fieldsName = ["VOLUME", "TTGP", "PFV", "Vba", "Distance_to_Glacier", "Glacier Volume", "Slope",
                      "probability_To_Channel", "IC_Mean", "Pre", "Tem"]
        list_label = ["Target"]
        list_fieldNames = ["VOLUME", "TTGP", "PFV", "Vba", "Distance_to_Glacier", "Glacier Volume", "Slope",
                      "probability_To_Channel", "IC_Mean", "Pre", "Tem","Target"]
        df_input = pd.read_excel("input/Yangben_GlacialDF.xlsx")
        filePath = "output/Yangben_GlacialDF_SMOTE.xlsx"
        # 获取矩阵
        import ExcelControl as EXLctl
        matrix_X = EXLctl.convertDFtoArray(df_input, fieldsName)
        matrix_Y = EXLctl.convertDFtoArray(df_input, list_label)
        do_RandomOverSampler(matrix_X, matrix_Y, list_fieldNames, filePath)

    if flag_RandomUnderSampler:
        fieldsName = ["HeightDi_1","Channel__1","Rock_T_1","Dis_Faul_1","E_Volume_1","Landcove_1","NDVI_1","F25num","DayMax_1"]
        list_label = ["Target"]
        list_fieldNames = ["HeightDi_1","Channel__1","Rock_T_1","Dis_Faul_1","E_Volume_1","Landcove_1","NDVI_1","F25num","DayMax_1","Target"]
        # df_input = pd.read_excel("input/Results_RO_ZB.xlsx")
        df_input = pd.read_excel("input/Yangben_Nepel_ZB_Current.xlsx")

        filePath = "output/Yangben_Nepel_ZB_Current_US.xlsx"
        # 获取矩阵
        import ExcelControl as EXLctl
        matrix_X = EXLctl.convertDFtoArray(df_input, fieldsName)
        matrix_Y = EXLctl.convertDFtoArray(df_input, list_label)
        do_RandomUnderSampler(matrix_X, matrix_Y, list_fieldNames, filePath)