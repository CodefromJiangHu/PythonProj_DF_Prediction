import numpy as np
from sklearn.metrics import roc_curve, auc, mean_squared_error  ###计算roc和auc
from ROC_Fig import acu_curve, save, cross_validation_SVC_plot_ROC, writeMessage
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import math
import PlotControl as plt_ctl

# 对于部分只能输出0-1连续结果的模型，采用auc来确定阈值，并进行分类结果划分
def get_results_classification_lstm(model, X_test):
    prepro = np.array(model.predict(X_test))
    pre_results = np.argmax(prepro, axis=1)
    # prepro = prepro[:, 1]
    # fpr, tpr, thresholds = roc_curve(Y_test, prepro)  ###计算真阳性率和假阳性率
    # # 确定auc最大处的阈值
    # idx = np.linalg.norm(
    #     (np.array([[0, 1]]) - np.stack([fpr, tpr], axis=1)),
    #     axis=1).argmax()
    # max_thresh = thresholds[idx]
    # pre_results = np.where(prepro>max_thresh,1,0)
    return pre_results

def get_acc_with_lstm(_model, X_test, Y_test):
    model = _model
    Y_test = Y_test[:, 1]
    # 获取二分类结果
    prepro = get_results_classification_lstm(model, X_test)
    # 计算交叉熵损失
    acc = accuracy_score(Y_test, prepro)
    return acc

def post_date_processing_lstm(_model, modelNmae, X_test, Y_test,  X_total_std, str_para_vary, flagSimle, df_totalyangben_environment):
    model = _model
    Y_test = Y_test[:, 1]
    # 获取二分类结果
    prepro = get_results_classification_lstm(model, X_test)
    # 计算交叉熵损失
    loss = log_loss(Y_test, prepro)
    acc = accuracy_score(Y_test, prepro)
    print("acc:" + str(acc))
    # 调试结果记录
    writeMessage("最优参数："+str_para_vary , "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("交叉熵损失和ACC：" + str(loss) + "," + str(acc), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    # 2.样本集:预测精度打印输出 ROC AUC(模型预测性能)
    # 样本集 ： 预测结果验证
    prepro_yangben = model.predict(X_test)
    prepro_yangben = np.array(prepro_yangben[:, 1])
    # fpr 与 tpr 输出
    # acu_curve(y_taget, prepro_yangben)  # ROC曲线绘制
    fpr, tpr, thresholds = roc_curve(Y_test, prepro_yangben)  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    print("AUC", roc_auc)
    # output auc value in to txt
    writeMessage("AUC：" + str(roc_auc), "outputs/" + modelNmae + "_优化器参数调试结果.txt")

    # calculate mse and rmse
    mse = mean_squared_error(Y_test, prepro_yangben)
    rmse = math.sqrt(mse)
    writeMessage("MSE：" + str(mse), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("RMSE：" + str(rmse), "outputs/" + modelNmae + "_优化器参数调试结果.txt")


    TEMLIST = np.array((fpr, tpr), dtype='float32')
    if not (flagSimle):
        save(TEMLIST, "outputs/" + str(str_para_vary) + modelNmae + "_yangben_ROC_resuts.xlsx")
    # 全流域趋向泥石流概率计算
    # 计时开始
    # start_time = datetime.datetime.now()
    result_pro = model.predict(X_total_std)
    result_pro = result_pro[:,1]
    # result = model.predict(X_total_std)
    # 计时结束
    # end_time = datetime.datetime.now()
    # time_dif = get_timedif_seconds(start_time, end_time)
    result_pro_1 = np.array(result_pro)
    # 结果保存
    df_totalyangben_environment['result_pro'] = result_pro_1
    # df_totalyangben_environment['result'] = result
    # 将结果保存到第二个sheet里面
    if not (flagSimle):
        df_totalyangben_environment.to_excel("outputs/" + str(str_para_vary) + modelNmae + "易发性_results.xlsx",
                                             sheet_name="Resutls",
                                             index=False, header=True)
    print("评估结束！")

# just for prediction operation
def post_date_processing_lstm_for_predic(_model, modelNmae, X_total_std, str_para_vary, flagSimle, df_totalyangben_environment):
    model = _model
    # run predict
    result_pro = model.predict(X_total_std)
    result_pro = result_pro[:,1]
    result_pro_1 = np.array(result_pro)
    # saving results
    df_totalyangben_environment['result_pro'] = result_pro_1
    if not (flagSimle):
        df_totalyangben_environment.to_excel("outputs/" + str(str_para_vary) + modelNmae + "_sus_results.xlsx",
                                             sheet_name="Resutls",
                                             index=False, header=True)
    print("Prediction over！")

def post_date_processing_lstm_gsd(_model, modelNmae, X_test, Y_test,  X_total_std, str_para_vary, flagSimle, df_totalyangben_environment,list_label,tool_for_std):
    model = _model
    # Y_test = Y_test[:, 1]
    # 获取预测的土体强度参数，C和Phi的标准化值
    test_Predict = model.predict(X_test)

    TestMSE = mean_squared_error(Y_test, test_Predict)
    TestRMSE = math.sqrt(mean_squared_error(Y_test, test_Predict))
    r2_Test = r2_score(Y_test, test_Predict)
    print('Test Score: %.2f MSE' % (TestMSE))
    print('Test Score: %.2f RMSE' % (TestRMSE))
    print('Test R2: %.2f' % (r2_Test))

    # 调试结果记录
    writeMessage("最优参数："+str_para_vary , "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("Test MSE:" + str(TestMSE), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("Test RMSE:" + str(TestRMSE) , "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("Test R2: " + str(r2_Test), "outputs/" + modelNmae + "_优化器参数调试结果.txt")

    # 全流域趋向泥石流概率计算
    # 计时开始
    # start_time = datetime.datetime.now()
    result_pre = model.predict(X_total_std)
    result_pre = np.array(result_pre)
    #
    all_results = np.hstack((X_total_std[1], result_pre))
    standarded_Matrix = all_results
    inverse_standarded_Matrix = tool_for_std.inverse_transform(standarded_Matrix)
    inverse_result_pre_Y = inverse_standarded_Matrix[:, -1:]
    # 结果保存
    df_totalyangben_environment[list_label] = inverse_result_pre_Y
    # df_totalyangben_environment[list_label[1]] = inverse_result_pre_Y[:, 1]
    # 将结果保存到第二个sheet里面
    if not (flagSimle):
        df_totalyangben_environment.to_excel("outputs/" + str(str_para_vary) + modelNmae + "参数估算_results.xlsx",
                                             sheet_name="Resutls",
                                             index=False, header=True)
    print("评估结束！")

# just for prediction operation
def post_date_processing_for_predic(_model, modelNmae, X_total_std, str_para_vary, flagSimle, df_totalyangben_environment):
    model = _model
    # run predict
    result_pro = model.predict_proba(X_total_std)
    result = model.predict(X_total_std)
    result_pro_1 = np.array(result_pro[:, 1])
    # saving results
    df_totalyangben_environment['result_pro'] = result_pro_1
    # df_totalyangben_environment['result'] = result
    if not (flagSimle):
        df_totalyangben_environment.to_excel("outputs/" + str(str_para_vary) + modelNmae + "_sus_results.xlsx",
                                             sheet_name="Resutls",
                                             index=False, header=True)
    print("Prediction over！")

def post_date_processing(_model, modelNmae, X_test, Y_test, X_yangben_std, Y_yangben, X_total_std, str_para_vary, flagSimle,df_totalyangben_environment):
    model = _model
    #  返回损失值
    prepro = model.predict(X_test)
    # 计算交叉熵损失
    loss = log_loss(Y_test, prepro)
    acc = accuracy_score(Y_test, prepro)
    print("acc:" + str(acc))
    # 调试结果记录
    writeMessage("最优参数："+str_para_vary , "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("交叉熵损失和ACC：" + str(loss) + "," + str(acc), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    # 2.样本集:预测精度打印输出 ROC AUC(模型预测性能)
    # 样本集 ： 预测结果验证
    prepro_yangben = model.predict_proba(X_test)
    prepro_yangben = np.array(prepro_yangben)
    # fpr 与 tpr 输出
    # acu_curve(y_taget, prepro_yangben)  # ROC曲线绘制
    fpr, tpr, threshold = roc_curve(Y_test, prepro_yangben[:, 1])  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    # output auc value in to txt
    writeMessage("AUC：" + str(roc_auc), "outputs/" + modelNmae + "_优化器参数调试结果.txt")

    # calculate mse and rmse
    mse = mean_squared_error(Y_test, prepro_yangben[:, 1])
    rmse = math.sqrt(mse)
    writeMessage("MSE：" + str(mse), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("RMSE：" + str(rmse), "outputs/" + modelNmae + "_优化器参数调试结果.txt")


    TEMLIST = np.array((fpr, tpr), dtype='float32')
    if not (flagSimle):
        save(TEMLIST, "outputs/" + modelNmae + str(str_para_vary) + "_yangben_ROC_resuts.xlsx")
    # 全流域趋向泥石流概率计算
    # 计时开始
    # start_time = datetime.datetime.now()
    result_pro = model.predict_proba(X_total_std)
    result = model.predict(X_total_std)
    # 计时结束
    # end_time = datetime.datetime.now()
    # time_dif = get_timedif_seconds(start_time, end_time)
    result_pro_1 = np.array(result_pro[:, 1])
    # 结果保存
    df_totalyangben_environment['result_pro'] = result_pro_1
    df_totalyangben_environment['result'] = result
    # 将结果保存到第二个sheet里面
    if not (flagSimle):
        df_totalyangben_environment.to_excel("outputs/" + modelNmae + str(str_para_vary) + "_sus_results.xlsx",
                                             sheet_name="Resutls",
                                             index=False, header=True)
    print("评估结束！")


def post_date_processing_regressor(_model, modelNmae, X_test, Y_test, X_yangben_std, Y_yangben, X_total_std, str_para_vary, flagSimle,df_totalyangben_environment,field_name_for_opt):
    model = _model
    #  返回损失值

    # 调试结果记录
    writeMessage("最优参数："+str_para_vary , "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    # 2.样本集:预测精度打印输出 ROC AUC(模型预测性能)
    # 样本集 ： 预测结果验证
    # prepro_yangben = model.predict(X_test)
    # prepro_yangben = np.array(prepro_yangben)
    #
    # # calculate mse and rmse
    # mse = mean_squared_error(Y_test.flatten(), prepro_yangben)
    # rmse = math.sqrt(mse)
    # 训练集
    train_Predict=model.predict(X_yangben_std)
    TrainMSE = mean_squared_error(Y_yangben, train_Predict)
    TrainRMSE = math.sqrt(mean_squared_error(Y_yangben, train_Predict))
    plt_ctl.plot_scater_line_for_predict(Y_yangben, train_Predict)
    # 测试集验证精度输出
    r2_Train = r2_score(Y_yangben, train_Predict)
    print('Train Score: %.2f MSE' % (TrainMSE))
    print('Train Score: %.2f RMSE' % (TrainRMSE))
    print('Train R2: %.2f' % (r2_Train))
    writeMessage("Train MSE：" + str(TrainMSE), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("Train RMSE：" + str(TrainRMSE), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("Train R2：" + str(r2_Train), "outputs/" + modelNmae + "_优化器参数调试结果.txt")

    # 测试集
    test_Predict = model.predict(X_test)
    TestMSE = mean_squared_error(Y_test, test_Predict)
    TestRMSE = math.sqrt(mean_squared_error(Y_test, test_Predict))
    # 绘制点线图
    plt_ctl.plot_scater_line_for_predict(Y_test, test_Predict)
    # 测试集验证精度输出
    r2_Test = r2_score(Y_test, test_Predict)
    print('Test Score: %.2f MSE' % (TestMSE))
    print('Test Score: %.2f RMSE' % (TestRMSE))
    print('Test R2: %.2f' % (r2_Test))

    writeMessage("Test MSE：" + str(TestMSE), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("Test RMSE：" + str(TestRMSE), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
    writeMessage("Test R2：" + str(r2_Test), "outputs/" + modelNmae + "_优化器参数调试结果.txt")

    # 全流域趋向泥石流概率计算
    # 计时开始
    # start_time = datetime.datetime.now()
    result = model.predict(X_total_std)
    # 计时结束
    # end_time = datetime.datetime.now()
    # time_dif = get_timedif_seconds(start_time, end_time)
    # 结果保存
    df_totalyangben_environment[field_name_for_opt] = result
    # 将结果保存到第二个sheet里面
    if not (flagSimle):
        df_totalyangben_environment.to_excel("outputs/"  + modelNmae +"_" + str(str_para_vary)+ "_sus_results.xlsx",
                                             sheet_name="Resutls",
                                             index=False, header=True)
    print("评估结束！")



if __name__ == "__main__":
    #minmax_demo()
    standard_matrix()
