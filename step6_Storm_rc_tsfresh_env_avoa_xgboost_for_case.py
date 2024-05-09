# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import numpy as np

from DataPostPrecessingControl import post_date_processing_for_predic
import DataPreDrocessingControl as dpdCL
import ExcelControl as excelCL

# 保存训练好的模型的库
import joblib as jblib

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

if __name__ == '__main__':

    # controlling flag
    flag_prediction = True
    flagSimle = False
    flag_fit_scaler = False

    # parameters detial info
    str_storm_name = "2022Storm_062006"
    modelNmae = str_storm_name+"_rc24_GroupB_AVOA_XGB"
    df_totalyangben_environment = pd.read_excel('input/Storm_DynamicAnalysis/step5p0_total_'+str_storm_name+'_RC_case.xlsx')

    # parameters detial info
    row_size = df_totalyangben_environment.shape[0]
    colum_size = 19

    # setting runtime name
    str_para_vary = "run_optimal_xgboost_"

    # reading samples
    list_fields_names = ["AE", "AEMR", "ATC", 'CQ1', 'CQ2',
                         "CJ", "RH", "AI",
                         "AP", "AT", "LT", "DF", "SE", "SD",
                         "NDVI", "LD", "RD", "PD", "EP72"]
    list_fields_names_plot = ["AE", "AEMR", "ATC", 'CQ1', 'CQ2',
                              "CG", "RH", "AI",
                              "AP", "AT", "LT", "DF", "SE", "SD",
                              "NDVI", "LD", "RD", "PD", "EP72"]
    list_label = ["Target"]

    X_total, Y_total = get_data_from_excel(df_totalyangben_environment, row_size, colum_size, list_fields_names,
                                           list_label, False)

    # (1)samples preprocessing
    saved_scaler_filepath = "saved_para/saved_scaler_1.0.pkl"
    if flag_fit_scaler:
        df_totalyangben_environment_forStandard = pd.read_excel('output/step5p0_total_RC_total.xlsx')
        X_total_for_sd, Y_total_for_sd = get_data_from_excel(df_totalyangben_environment_forStandard, row_size, colum_size, list_fields_names,
                                               list_label, False)
        sc_first = dpdCL.dataprocessing_first_getStandardScaler(X_total_for_sd)
        dpdCL.save_scaler(sc_first, saved_scaler_filepath)
    sc = dpdCL.load_scaler(saved_scaler_filepath)
    X_total_std = dpdCL.dataprocessing_second_StandardScaler_transform(sc, X_total)

    if flag_prediction:
        # obtaining trainted model
        model_saved_path = "saved_model/trained_model_xgboost_rc24_tsfresh.pkl"
        # model loading
        model = jblib.load(model_saved_path)
        # prediction and validation
        post_date_processing_for_predic(model, modelNmae,  X_total_std, str_para_vary,
                                        flagSimle, df_totalyangben_environment)


