import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
from matplotlib import pyplot as plt
# 设置显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体 
# plt.rcParams['axes.unicode_minus']=False     # 正常显示负号

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def cal_metrics(confusion_matrix,classes):
    metrics_result_dict = {x:[] for x in classes}    
    for index,label in enumerate(classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion_matrix)
        # 对角线上是正确预测的
        TP = confusion_matrix[index, index]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(confusion_matrix[:, index]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(confusion_matrix[index, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        # print(ALL,TP,TN,FP,FN)
        Accuracy = (TP+TN)/ALL
        Sensitivity = TP/(TP+FN)
        Specificity = TN/(TN+FP)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        #F-Measure是Precision和Recall加权调和平均
        F_measure = 2*Precision*Recall/(Precision+Recall)
        temp = np.array([Accuracy,Sensitivity,Specificity,Precision,Recall,F_measure])
        metrics_result_dict[label]= temp
    
    # metrics_result_dict["Macro-average"] = [key  for key in metrics_result_dict.keys() for i in  metrics_result_dict[key]]
    return metrics_result_dict

def plot_maxtrix(classes:list,y_true:list,y_pred:list,save_dir:str,acc:float,normalize=False):
    # 使用sklearn工具中confusion_matrix方法计算混淆矩阵
    confusion_mat = confusion_matrix(y_true, y_pred)
    # print("confusion_mat.shape : {}".format(confusion_mat.shape))
    # print("confusion_mat :\n {}".format(confusion_mat))
    #归一化
    if normalize:
        confusion_mat = confusion_mat.astype('float') /  confusion_mat.sum(axis=1)[:, np.newaxis]
        my_values_format = ".2f"
    else:
        my_values_format = "d"
        
    # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
    ###plot 1  
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,                        # 同上
        xticks_rotation="horizontal",   # 同上
        values_format=my_values_format               # 显示的数值格式
    )
    
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(confusion_mat, cmap=plt.cm.Blues)

    # plt.title('Confusion Matrix')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

    # tick_marks = np.arange(len(set(y_true)))
    # plt.xticks(tick_marks, set(y_true))
    # plt.yticks(tick_marks, set(y_true))

    # for i in range(confusion_mat.shape[0]):
    #     for j in range(confusion_mat.shape[1]):
    #         ax.text(x=j, y=i, s=confusion_mat[i, j], va='center', ha='center')
    # plt.show()
    str_acc = "%.3f %%"%acc
    plt.title("swin_large_patch4_window7_224 ce loss Confusion Matrix acc:%s"%(str_acc),loc="right")
    # plt.title("Confusion Matrix acc:%s"%(acc))
    fig_save_path = save_dir+"/save_fig.png"
    plt.savefig(fig_save_path,dpi=300)
    plt.close()
    # print("save_fig in %s"%(fig_save_path))
    """
    - 真阳性（True Positive，TP）：指被分类器正确分类的正例数据
    - 真阴性（True Negative，TN）：指被分类器正确分类的负例数据
    - 假阳性（False Positive，FP）：被错误地标记为正例数据的负例数据
    - 假阴性（False Negative，FN）：被错误地标记为负例数据的正例数据
    """
    performance = cal_metrics(confusion_mat,classes)##返回每个类别的各项指标以及综合的
    performance["Macro-average"]=np.array([i for i in performance.values()]).mean(axis=0)
    performance["Macro-average"][0] = acc/100
    # print(performance)
    #列名
    col=["Accuracy" ,"Sensitivity" ,"Specificity" ,"Precision","Recall","F_measure"]
    #行名
    row=[key for key in performance.keys()]
    vals = [i for i in performance.values()]
    plt.figure(figsize=(10,5))
    tab = plt.table(cellText=vals, #简单理解为表示表格里的数据
                # colWidths=[0.3]*6,     #每个小格子的宽度 * 个数，要对应相应个数
              colLabels=col, #每列的名称
             rowLabels=row,#每行的名称（从列名称的下一行开始）
              loc='center', #表格所在位置
              cellLoc='center',#列名称的对齐方式
              rowLoc='center',#行名称的对齐方式
              )
    tab.scale(1,3) 
    plt.axis('off')
    tab_save_path = fig_save_path.replace("fig","tab")
    plt.savefig(tab_save_path,dpi=300)
    plt.close()
    
    
    result = classification_report(y_true,y_pred,target_names=classes)
    acc = accuracy_score(y_true,y_pred)
    # print("result:\n",result,"\n acc :\n",acc)
    
# acc = "%.3f %%"%(100 * 12 / 100)
# acc = 0.6
# classes = ["00_NIML", "01_ASC-US", "02_LSIL", "03_ASC-H", "04_HSIL"]
# plot_maxtrix(classes,[1,2,3,4,5,5,2,1,3,4],[1,2,3,4,5,1,2,3,4,5],"./",acc)




