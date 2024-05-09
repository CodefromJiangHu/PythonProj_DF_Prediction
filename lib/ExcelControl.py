#!/usr/bin/python
# encoding: utf-8
import pandas as pd
import numpy as np
import os
import math

def meanValueForList(numberList):
    sum_value = sum(numberList)
    avg = sum_value/len(numberList)
    return avg



def floatParseFromStr(StrList):
    temList = []
    for index in range(len(StrList)):
        temList.append(float(StrList[index]))
    return temList
# 根据字段获取列数据，以逗号隔开
def get_listStr_from_df(df_excel,filedName):
    list_str = ""
    list  = df_excel[filedName].tolist()
    for i in range(len(list)):
        if i ==0:
            list_str = str(list[i])
        else:
            list_str = list_str + ","+str(list[i])
    return list_str
# 根据excel的主键，获取相应字段对应的值
def getValueByKeyValue(base_df, key_fieldName, key_value, return_filedName):
    result = ""
    df = base_df
    df_key = df[df[key_fieldName] == key_value]
    resultsList = df_key[return_filedName].tolist()
    if(len(resultsList)<1):
        result = "无"
    else:
        result = str(resultsList[0])
    return result
# 根据以逗号为分隔符号的降雨序列，获取平均雨强序列
def getAvgRainfallIntenseByList(StrList):
    AvgRainfallList = []
    for index in range(len(StrList)):
        rainfallList = StrList[index].split(',')
        tem_rainfallList = floatParseFromStr(rainfallList)
        avg_value = meanValueForList(tem_rainfallList)
        AvgRainfallList.append( avg_value)
    return AvgRainfallList

def get_effective_prec_before(StrList,interval_hours,effective_constant,nodata_value):
    rainfallList = StrList.split(',')
    effctive_rainrall = 0.0
    tem_rainfallList = floatParseFromStr(rainfallList)
    sum_rainfall = 0
    days = 0
    times = int(len(tem_rainfallList)/interval_hours)
    for i in  range(len(tem_rainfallList)):
        if tem_rainfallList[i] == nodata_value:
            tem_rainfallList[i] = 0
        sum_rainfall = sum_rainfall+tem_rainfallList[i]
        if (i+1)%interval_hours == 0:
            effctive_rainrall= effctive_rainrall + sum_rainfall*math.pow(effective_constant, times-days)
            days = days + 1
            sum_rainfall = 0
    return effctive_rainrall


# 根据以逗号为分隔符号的降雨序列，获取平均雨强序列
def getAccumRainfallByList(StrList):
    AccumRainfallList = []
    for index in range(len(StrList)):
        rainfallList = StrList[index].split(',')
        tem_rainfallList = floatParseFromStr(rainfallList)
        sum_value = sum(tem_rainfallList)
        AccumRainfallList.append(sum_value)
    return AccumRainfallList

# 按字段名字顺序将df转化为矩阵:
def convertDFtoArray(input_df, list_fieldsName):
    array = input_df.values
    tem_array = np.array(array)
    m, n = tem_array.shape
    # tem_2dArray = np.zeros(shape=[m, 1])
    final_array = np.zeros(shape=[m, len(list_fieldsName)])
    for i in range(len(list_fieldsName)):
        tem = np.array(input_df[list_fieldsName[i]].values).reshape(m)
        final_array[:, i] = tem
    return final_array

def merge_multi_csv(Folder_Path,SaveFile_Path,SaveFile_Name):
    """
    :param Folder_Path: # 要拼接的文件夹及其完整路径，注意不要包含中文
    :param SaveFile_Path: # 拼接后要保存的文件路径
    :param SaveFile_Name: # 合并后要保存的文件名
    :return:
    """
    '''
    Data:2022-02-23
    Auther;Hu Jiang
    Description:使用Pandas拼接多个CSV文件到一个文件（即合并）
    '''
    #
    # import pandas as pd
    # import os
    # Folder_Path = r'C:\foldername'
    # SaveFile_Path = r'C:\foldername'
    # SaveFile_Name = r'all.csv'
    Folder_Path = Folder_Path
    SaveFile_Path = SaveFile_Path
    SaveFile_Name = SaveFile_Name
    # # 修改当前工作目录
    # os.chdir(Folder_Path)
    # 将该文件夹下的所有文件名存入一个列表
    file_list = os.listdir(Folder_Path)
    # 读取第一个CSV文件并包含表头
    df = pd.read_csv(Folder_Path + '\\' + file_list[0])  # 编码默认UTF-8，若乱码自行更改
    # 将读取的第一个CSV文件写入合并后的文件保存
    df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)
    # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
    for i in range(1, len(file_list)):
        df = pd.read_csv(Folder_Path + '\\' + file_list[i])
        df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')
def merge_multi_excel(Folder_Path,SaveFile_Path,SaveFile_Name):
    """
    :param Folder_Path: # 要拼接的文件夹及其完整路径，注意不要包含中文
    :param SaveFile_Path: # 拼接后要保存的文件路径
    :param SaveFile_Name: # 合并后要保存的文件名
    :return:
    """
    '''
    Data:2022-02-23
    Auther;Hu Jiang
    Description:使用Pandas拼接多个CSV文件到一个文件（即合并）
    '''
    #
    # import pandas as pd
    # import os
    # Folder_Path = r'C:\foldername'
    # SaveFile_Path = r'C:\foldername'
    # SaveFile_Name = r'all.csv'
    Folder_Path = Folder_Path
    SaveFile_Path = SaveFile_Path
    SaveFile_Name = SaveFile_Name
    # # 修改当前工作目录
    # os.chdir(Folder_Path)
    # 将该文件夹下的所有文件名存入一个列表
    file_list = os.listdir(Folder_Path)
    df_list = []
    # 读取第一个CSV文件并包含表头
    df = pd.read_excel(Folder_Path + '\\' + file_list[0])  # 编码默认UTF-8，若乱码自行更改

    df_list.append(df)
    # 将读取的第一个CSV文件写入合并后的文件保存
    df.to_excel(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)
    # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
    for i in range(1, len(file_list)):
        df = pd.read_excel(Folder_Path + '\\' + file_list[i])
        df_list.append(df)
    # 3.使用pd.concat进行df批量合并
    df_merged = pd.concat(df_list)
    df_merged.to_excel(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)
# 提供需要融合的excel文件名数组，然后实施融合操作
def merge_multi_excel_with_filenames(Folder_Path,filename_list,SaveFile_Path,SaveFile_Name):
    """
    :param Folder_Path: # 要拼接的文件夹及其完整路径，注意不要包含中文
    :param SaveFile_Path: # 拼接后要保存的文件路径
    :param SaveFile_Name: # 合并后要保存的文件名
    :return:
    """
    '''
    Data:2022-02-23
    Auther;Hu Jiang
    Description:使用Pandas拼接多个CSV文件到一个文件（即合并）
    '''
    #
    # import pandas as pd
    # import os
    # Folder_Path = r'C:\foldername'
    # SaveFile_Path = r'C:\foldername'
    # SaveFile_Name = r'all.csv'
    Folder_Path = Folder_Path
    SaveFile_Path = SaveFile_Path
    SaveFile_Name = SaveFile_Name
    # # 修改当前工作目录
    # os.chdir(Folder_Path)
    # 将该文件夹下的所有文件名存入一个列表
    file_list = filename_list
    df_list = []
    # 读取第一个CSV文件并包含表头
    df = pd.read_excel(Folder_Path + '\\' + file_list[0])  # 编码默认UTF-8，若乱码自行更改

    df_list.append(df)
    # 将读取的第一个CSV文件写入合并后的文件保存
    df.to_excel(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)
    # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
    for i in range(1, len(file_list)):
        df = pd.read_excel(Folder_Path + '\\' + file_list[i])
        df_list.append(df)
    # 3.使用pd.concat进行df批量合并
    df_merged = pd.concat(df_list)
    df_merged.to_excel(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)

# 根据字段名称和矩阵构建DF
def construct_DF(list_keys, matrix_values):
    """
    list_keys = ["a", "b", "c","d","e"]
    tem_matrix = [[1.0, 4.0, 1.0, 1.0, 1.0],
                  [1, 2, 8, 5, 1],
                  [1, 3, 7, 4, 1],
                  [1, 3, 4, 6, 1],
                  [1.0, 3, 1, 1, 1.0]]
    tem_matrix = np.array(tem_matrix)
    df = construct_DF(list_keys, tem_matrix)
    df.to_excel("test.xlsx")
    """
    list_values = np.transpose(matrix_values)
    dict = {}
    # 定义连个列表
    list_keys = list_keys
    list_values = list_values

    # 遍历两个列表，构造字典键值对并加入字典
    for i, j in zip(list_keys, list_values):
        dict.update({i: j})
    df = pd.DataFrame(data=dict)
    return df
# 更新字段名称
def updating_fieldnames(df_excel, old_names,new_names,optfilePath):
    # 读取 Excel 文件
    # df = pd.read_excel('your_excel_file.xlsx', sheet_name='your_sheet_name')
    df = df_excel
    # 将要替换的列名称存储在一个列表中
    # old_names = ['A', 'B']
    # new_names = ['New Name A', 'New Name B']
    # 使用列表推导式遍历旧列名和新列名，并用新的列名称替换旧的列名称
    df.columns = [new_names[old_names.index(col)] if col in old_names else col for col in df.columns]
    # 将修改后的数据写入一个新的 Excel 文件
    df.to_excel(optfilePath, index=False)


if __name__ == '__main__':



    flag_merge_csv = False
    flag_merge_excel = True
    if flag_merge_csv:
        Folder_Path = r'input\csvfiles'
        SaveFile_Path = r'output\csv_merge'
        SaveFile_Name = r'csv_all.csv'
        merge_multi_csv(Folder_Path, SaveFile_Path, SaveFile_Name)
    if flag_merge_excel:
        Folder_Path = r'input\excelfiles'
        SaveFile_Path = r'output\excel_merge'
        SaveFile_Name = r'excel_all.xlsx'
        merge_multi_excel(Folder_Path, SaveFile_Path, SaveFile_Name)


