import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from openpyxl import Workbook
# 绘制ROC曲线
from sklearn import metrics
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt
# 训练集交叉验证
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
from sklearn.metrics import RocCurveDisplay
# from CrossValidation_Plot_Fig import get_groups
import datetime
import os
from sklearn.metrics import zero_one_loss
from sklearn.metrics import log_loss
# 准确率统一计算 accuracy = (TP + TN)/(TP + FP + TN + FN)
from sklearn.metrics import accuracy_score



#method

def writeMessage(message_txt,txt_filePathName):
    filepath = txt_filePathName # ‘xxx.txt’
    if os.path.exists(filepath) == False:
        # 创建文件
        file = open(filepath, 'w', encoding='utf-8')
        file.close()
    file = open(filepath, 'r+', encoding='utf-8')
    # 读取一下数据，调整指针到末尾
    file.read()
    file.write(message_txt + "; \n")
    file.close()


def save(data, path):
    wb = Workbook()
    ws = wb.active # 激活 worksheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i, j])
        ws.append(row)
    wb.save(path)


def acu_curve(y, prob):
    # y真实prob预测
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.show()

def cross_validation_SVM_plot_ROC(mode_classifier, X_train, Y_train, _n_splits):
    """
    :param mode_classifier: 拟需要拟合的模型
    :param X_train: 训练集 的特征矩阵
    :param Y_train: 训练集 的标签矩阵
    :param _n_splits: 交叉验证划分的块数
    :return:
    """
    cv = StratifiedKFold(n_splits=_n_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    # 计算ROC 横纵坐标
    for i, (train, test) in enumerate(cv.split(X_train, Y_train)):
        mode_classifier.fit(X_train[train], Y_train[train])
        viz = RocCurveDisplay.from_estimator(
            mode_classifier,
            X_train[test],
            Y_train[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    # 显示样式设置
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    # 求解ROC均线的tpr值
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.savefig("cross_validation_plot_roc.png", bbox_inches="tight")
    plt.show()
    return mode_classifier
def cross_validation_CNN_AllImage_plot_ROC(mode_classifier, X_train, Y_train, _n_splits,cv_StratifiedShuffleSplit,_groups):
    """
    :param mode_classifier: 拟需要拟合的模型
    :param X_train: 训练集 的特征矩阵
    :param Y_train: 训练集 的标签矩阵
    :param _n_splits: 交叉验证划分的块数
    :return: model
    """
    # cv = StratifiedKFold(n_splits=_n_splits, random_state=6)
    cv = cv_StratifiedShuffleSplit
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    # 计算ROC 横纵坐标
    for i, (train, test) in enumerate(cv.split(X_train, Y_train, groups=_groups)):
        # mode_classifier.fit(X_train[train], Y_train[train])
        mode_classifier.fit(X_train[train], Y_train[train], validation_split=0.0, batch_size=5, epochs=1)
        # viz = RocCurveDisplay.from_predictions(
        #     mode_classifier,
        #     [X_train[test], X_train_auxiliary[test]],
        #     Y_train[test],
        #     # name="ROC fold {}".format(i),
        #     # alpha=0.3,
        #     # lw=1,
        #     # ax=ax,
        # )
        tem_X_train = np.array(X_train[test]).astype("float")
        # tem_X_train_auxiliary = np.array(X_train_auxiliary[test]).astype("float")
        prepro_yangben = mode_classifier.predict(tem_X_train)
        prepro_yangben = np.array(prepro_yangben)
        # 获取fpr tpr roc
        fpr, tpr, thresholds = roc_curve(Y_train[test], prepro_yangben)
        roc_auc = metrics.auc(fpr, tpr)
        viz = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        viz.name = "ROC fold {}".format(i)
        viz.alpha = 0.3
        # viz.lw = 1
        viz.ax_ = ax

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    # 显示样式设置
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    # 求解ROC均线的tpr值
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.savefig("cross_validation_plot_roc.png", bbox_inches="tight")
    plt.show()
    return mode_classifier

def cross_validation_CNN_plot_ROC(mode_classifier, X_train, X_train_auxiliary,Y_train, _n_splits,cv_StratifiedShuffleSplit,_groups,para_vary):
    """
    :param mode_classifier: 拟需要拟合的模型
    :param X_train: 训练集 的特征矩阵
    :param Y_train: 训练集 的标签矩阵
    :param _n_splits: 交叉验证划分的块数
    :return: model
    """
    # cv = StratifiedKFold(n_splits=_n_splits, random_state=6)
    cv = cv_StratifiedShuffleSplit
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    # 计算ROC 横纵坐标
    for i, (train, test) in enumerate(cv.split(X_train, Y_train, groups=_groups)):
        # mode_classifier.fit(X_train[train], Y_train[train])
        mode_classifier.fit([X_train[train], X_train_auxiliary[train]], Y_train[train], validation_split=0.0, batch_size = 5, epochs=1)
        # 求解每次拟合的精度
        #  返回损失值
        loss, acc = mode_classifier.evaluate([X_train[test], X_train_auxiliary[test]],  Y_train[test])
        # 损失 精度 输出
        tem_msg = "ROC fold {}".format(i) + ": [loss, accuracy] = " + str(loss)+","+str(acc)
        writeMessage(str(datetime.datetime.now()), "损失与精度输出"+str(para_vary)+".txt")
        writeMessage(tem_msg, "损失与精度输出"+str(para_vary)+".txt")
        
        # 求解ROC曲线的坐标
        tem_X_train = np.array(X_train[test]).astype("float")
        tem_X_train_auxiliary = np.array(X_train_auxiliary[test]).astype("float")
        prepro_yangben = mode_classifier.predict([tem_X_train, tem_X_train_auxiliary])
        prepro_yangben = np.array(prepro_yangben)
        # 获取fpr tpr roc
        fpr, tpr, thresholds = roc_curve(Y_train[test], prepro_yangben)
        roc_auc = metrics.auc(fpr, tpr)
        viz = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        # 绘制ROC曲线
        ax.plot(
            fpr,
            tpr,
            # color="b",
            # label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            label="ROC fold {}".format(i),
            lw=1,
            alpha=0.3,
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    # 显示样式设置
    # 绘制随机线
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    # 求解ROC均线的tpr值
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    # 绘制区间填充
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    # 设置图片坐标轴上下限
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic curves",
    )
    ax.legend(loc="lower right")
    # 保存图片

    plt.savefig("_cross_validation_plot_roc"+str(para_vary)+".png", bbox_inches="tight")
    # plt.show()
    return mode_classifier
def cross_validation_SVC_plot_ROC(mode_classifier, X_train, Y_train, _n_splits,cv_StratifiedShuffleSplit,_groups, para_vary,_flag_simple):
    """
        :param mode_classifier: 拟需要拟合的模型
        :param X_train: 训练集 的特征矩阵
        :param Y_train: 训练集 的标签矩阵
        :param _n_splits: 交叉验证划分的块数
        :return: model
        """
    flag_simple = _flag_simple
    # cv = StratifiedKFold(n_splits=_n_splits, random_state=6)
    cv = cv_StratifiedShuffleSplit()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 计算ROC 横纵坐标
    # for i, (train, test) in enumerate(cv.split(X_train, Y_train, groups=_groups)):
    for i, (train, test) in enumerate(cv.split(X_train, Y_train.ravel())):
        # mode_classifier.fit(X_train[train], Y_train[train])
        mode_classifier.fit(X_train[train], Y_train[train])

        if flag_simple:
            continue
        # 求解每次拟合的精度
        #  返回损失值
        # prepro = mode_classifier.predict(X_train[test])
        # # 计算交叉熵损失
        # loss = log_loss(Y_train[test], prepro)
        # acc = accuracy_score(Y_train[test], prepro)

        prepro = mode_classifier.predict(X_train[test])
        # 计算交叉熵损失
        loss = log_loss(Y_train[test], prepro)
        acc = accuracy_score(Y_train[test], prepro)


        # acc = mode_classifier.score(X_train[test], Y_train[test])
        # 损失 精度 输出///////////////////////////////////////////////////////////////////////////
        loss_acc_txt_filePath = "outputs/损失与精度输出" + str(para_vary) + ".txt"
        tem_msg = "ROC fold {}".format(i) + ": [loss, accuracy] = " + str(loss) + "," + str(acc)
        # writeMessage(str(datetime.datetime.now()),loss_acc_txt_filePath)
        writeMessage(tem_msg, loss_acc_txt_filePath)

    if not (flag_simple):
        fig, ax = plt.subplots()
        # 求解ROC曲线的坐标
        tem_X_train = np.array(X_train[test]).astype("float")
        # tem_X_train_auxiliary = np.array(X_train_auxiliary[test]).astype("float")
        prepro_yangben = mode_classifier.predict_proba(tem_X_train)
        prepro_yangben = np.array(prepro_yangben)
        # 获取fpr tpr roc
        fpr, tpr, thresholds = roc_curve(Y_train[test], prepro_yangben[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        viz = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        # 绘制ROC曲线
        ax.plot(
            fpr,
            tpr,
            # color="b",
            # label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            label="ROC fold {}".format(i),
            lw=1,
            alpha=0.3,
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # 显示样式设置
        # 绘制随机线
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
        # 求解ROC均线的tpr值
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        # 绘制区间填充
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        # 设置图片坐标轴上下限
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic curves",
        )
        ax.legend(loc="lower right")
        # 保存图片

        plt.savefig("outputs/_cross_validation_plot_roc" + str(para_vary) + ".png", bbox_inches="tight")
        plt.show()
        # 释放资源
        plt.close()
    return mode_classifier


if __name__ == '__main__':
    # 数据准备阶段
    np.random.seed(1338)
    # Generate the class/group data
    X = np.random.randn(100, 10)
    percentiles_classes = [0.5, 0.5]
    y = np.hstack([[ii] * int(100 * perc)
                   for ii, perc in enumerate(percentiles_classes)])
    # 数据前处理
    svcmodel = SVC(C=0.8, kernel='linear', decision_function_shape='ovo', probability=True)
    # 交叉验证方式
    cv = StratifiedShuffleSplit
    this_cv = cv(n_splits=5, random_state=6)
    # 拟划分的组数
    num_groups = 10
    groups = get_groups(len(X), num_groups)
    cross_validation_SVC_plot_ROC(svcmodel, X, y, 5, this_cv, groups)



