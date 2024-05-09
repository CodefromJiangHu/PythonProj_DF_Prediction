import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
# heatMap相关库
import seaborn as sns

def plot_scater_line_for_predict(observed,predicted):
    # Create a DataFrame
    df = pd.DataFrame({
        'Observed': observed.flatten(),
        'Predicted': predicted.flatten()
    })

    # Plotting the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Observed'], df['Predicted'], alpha=0.7, label='Observed vs Predicted')

    # Plotting the line of perfect prediction
    line_start = min(df['Observed'].min(), df['Predicted'].min())
    line_end = max(df['Observed'].max(), df['Predicted'].max())
    plt.plot([line_start, line_end], [line_start, line_end], 'k--', label='Perfect Prediction Line')

    # Labeling the plot
    plt.xlabel('Observed Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Observed vs Predicted Values')
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

def plot_line_with_colorfulType(list_x,list_y):
    x = list_x
    y = list_y
    dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    fig.colorbar(line, ax=axs[0])

    # Use a boundary norm instead
    cmap = ListedColormap(['r', 'g', 'b'])
    norm = BoundaryNorm([0, 50, 75, 100], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    fig.colorbar(line, ax=axs[1])

    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_ylim(y.min()-10, y.min()+10)
    plt.show()


def plot_line_simple(list_x,list_y):
    # Data for plotting
    t = list_x
    s = list_y
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel='x_lable (unit)', ylabel='y_lable (unit)',
           title='title')
    ax.grid()
    # fig.savefig("test.png")
    plt.show()

def plot_heatMap_with_matrix(ipt_matrix,X_ticks,Y_ticks,opt_fileName,mask):
    cor = ipt_matrix
    cor = np.round(cor, 3)
    # 单元内字体参数
    font = {'family': 'Arial',
            'size': 14,
            }
    # 设置标签字体大小
    sns.set(font_scale=1.4)
    # 绘图的字体
    plt.rc('font', family='Arial')
    # 设置图框大小
    fig = plt.figure(figsize=(16, 12))
    # 设置子图1的位置
    ax1 = fig.add_subplot(1, 1, 1)
    # # 获取同样大小的0矩阵
    # mask = np.zeros_like(cor)
    # # x轴标签设置
    # # 生成左上角掩膜
    # # mask[np.triu_indices_from(mask)] = True
    # for i in range(len(mask)):
    #     for j in range(i + 1, len(mask[0])):
    #         mask[i][j] = True
    # 热力图参数设置
    sns.heatmap(cor, linewidths=0.05, ax=ax1, mask=mask, annot=True, annot_kws=font, vmax=1.0, vmin=-1.0, cmap='YlGnBu',
                center=0.5,
                cbar=True, robust=False)
    # 标题设置
    # ax1.set_title('User features - 01', fontdict=font)
    # ax1.set_xticklabels(X_ticks, rotation=0, fontsize='small')
    ax1.set_xticklabels(X_ticks, rotation=0, fontsize='small')
    ax1.set_yticklabels(Y_ticks, rotation=0, fontsize='small')
    plt.savefig(str(opt_fileName) + ".png", bbox_inches="tight")
    # plt.show()
# 绘制直方图
def plot_Histogram_frm_list(list_x, num_bins,label_x,label_y):
    x = list_x
    # num_bins = 50
    fig, ax = plt.subplots()
    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, density=True)
    ax.set_xlabel(label_x, fontsize=16)
    ax.set_ylabel(label_y, fontsize=16)
    # ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    flag_plot_line_with_colorfulType = False
    flag_plot_line_simple = True
    if flag_plot_line_with_colorfulType:
        x = np.linspace(0, 3 * np.pi, 500)
        y = np.sin(x)
        plot_line_with_colorfulType(x, y)
    if flag_plot_line_simple:
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        plot_line_simple(t, s)



