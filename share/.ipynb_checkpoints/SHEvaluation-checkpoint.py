'''
自动化训练分类模型（支持多分类），包括
1.模型训练
2.模型评估
3.ROC曲线绘制
'''
import h2o
import time,json,os,sys
import numpy as np
import pandas as pd
import pandas as pd
import json,os,sys,datetime,warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SHCommon import CSHCommon
from SHSample import CSHSample
from tqdm.notebook import tqdm
from IPython.display import display
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ORandomForestEstimator
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore") 
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)
'''
绘制ROC曲线，支持多分类（1 Vs 1 或 1 Vs Rest两种方式）
输入的数据格式为pd.DataFrame，以下为必须的字段
1.true : ground true
2.predict: predict reuslt
3.p0~pn: 不同分类对应的概率
'''
class CSHROC():
    
    def __init__(self,df_data):
        df_temp = df_data.copy(deep=True).reset_index(drop=True)
        self.m_true = df_temp['true']
        self.m_pred = df_temp['predict']
        classes = self.m_true.unique().tolist()
        prob_list = []
        for i in range(len(classes)):
            prob_list.append("p%d"%i)
        self.m_prob = df_temp[prob_list]

    def calculate_tpr_fpr(self,y_real, y_pred):
        cm = confusion_matrix(y_real, y_pred)
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
        # Calculates tpr and fpr
        tpr =  TP/(TP + FN) # sensitivity - true positive rate
        fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
        return tpr, fpr
 
    def get_prob_level(self,prob_list,prob_bins=100):
        ret = []
        ret.append(0)
        used_prob_bins = None
        unique_prob = list(set(prob_list))
        if len(unique_prob) <= prob_bins:
            used_prob_bins = unique_prob
        else:
            cat,used_prob_bins = pd.cut(prob_list,include_lowest=True,retbins=True,bins=prob_bins)
        for item in used_prob_bins:
            ret.append(item)
        ret.append(1)
        ret.sort()
        return list(set(ret))
   
    def get_all_roc_coordinates(self, y_real, y_proba,prob_bins = 100):
        '''
        Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.

        Args:
            y_real: The list or series with the real classes.
            y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

        Returns:
            tpr_list: The list of TPRs representing each threshold.
            fpr_list: The list of FPRs representing each threshold.
        '''
        tpr_list = [0]
        fpr_list = [0]
        threshold_list = self.get_prob_level(y_proba,prob_bins)
        for i in range(len(threshold_list)) :
            threshold = threshold_list[i]
            y_pred = y_proba >= threshold
            tpr, fpr = self.calculate_tpr_fpr(y_real, y_pred)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        return tpr_list, fpr_list
    
    def plot_roc_curve(self,tpr, fpr, scatter = True, ax = None):
        '''
        Plots the ROC Curve by using the list of coordinates (tpr and fpr).

        Args:
            tpr: The list of TPRs representing each coordinate.
            fpr: The list of FPRs representing each coordinate.
            scatter: When True, the points used on the calculation will be plotted with the line (default = True).
        '''
        if ax == None:
            plt.figure(figsize = (5, 5))
            ax = plt.axes()

        if scatter:
            sns.scatterplot(x = fpr, y = tpr, ax = ax)
        sns.lineplot(x = fpr, y = tpr, ax = ax)
        sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    
    # Plots the Probability Distributions and the ROC Curves One vs Rest
    def show_ROC_by_one_rest(self,bin_count = 20,prob_bins = 100 ,figsize = (20, 8)):
        plt.figure(figsize = figsize )
        bins = [i/bin_count for i in range(bin_count)] + [1]
        roc_auc_ovr = {}
        y_test = self.m_true.tolist()
        classes = self.m_true.unique().tolist()

        for i in range(len(classes)):
            c = classes[i]
            # Prepares an auxiliar dataframe to help with the plots
            df_aux = pd.DataFrame()
            df_aux['class'] = [1 if y == c else 0 for y in y_test]
            df_aux['prob'] = self.m_prob["p%d"%i]
            df_aux = df_aux.reset_index(drop = True)

            # Plots the probability distribution for the class and the rest
            ax = plt.subplot(2, len(classes), i+1)
            sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
            ax.set_title(c)
            ax.legend([f"Class: {c}", "Rest"])
            ax.set_xlabel(f"P(x = {c})")

            # Calculates the ROC Coordinates and plots the ROC Curves
            ax_bottom = plt.subplot(2, len(classes), i+len(classes)+1)
            tpr, fpr = self.get_all_roc_coordinates(df_aux['class'], df_aux['prob'],prob_bins)
            self.plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
            ax_bottom.set_title("ROC Curve %dvR"%c)

            # Calculates the ROC AUC OvR
            roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])

        # Displays the ROC AUC for each class
        avg_roc_auc = 0
        i = 0
        for k in roc_auc_ovr:
            avg_roc_auc += roc_auc_ovr[k]
            i += 1
            print(f"{k} ROC AUC OvR: {roc_auc_ovr[k]:.4f}")
        print(f"average ROC AUC OvR: {avg_roc_auc/i:.4f}")
        plt.tight_layout()
        plt.show()

    # Plots the Probability Distributions and the ROC Curves One vs Rest
    def show_ROC_by_one_one(self,bin_count = 20, prob_bins = 100, figsize = (20, 8)):
        classes_combinations = []
        classes = self.m_true.unique().tolist()
        class_list = list(classes)
        for i in range(len(class_list)):
            for j in range(i+1, len(class_list)):
                classes_combinations.append([class_list[i], class_list[j]])
                classes_combinations.append([class_list[j], class_list[i]])
        
        plt.figure(figsize = figsize )
        bins = [i/bin_count for i in range(bin_count)] + [1]
        roc_auc_ovo = {}

        for i in range(len(classes_combinations)):
            # Gets the class
            comb = classes_combinations[i]
            c1 = comb[0]
            c2 = comb[1]
            c1_index = class_list.index(c1)
            title = str(c1) + " vs " + str( c2 )

            # Prepares an auxiliar dataframe to help with the plots
            df_aux = pd.DataFrame()
            df_aux['class'] = self.m_true
            df_aux['prob'] = self.m_prob["p%d"%c1_index]

            # Slices only the subset with both classes
            df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
            df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
            df_aux = df_aux.reset_index(drop = True)

            # Plots the probability distribution for the class and the rest
            ax = plt.subplot(2, len(classes_combinations), i+1)
            sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
            ax.set_title(title)
            ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
            ax.set_xlabel(f"P(x = {c1})")

            # Calculates the ROC Coordinates and plots the ROC Curves
            ax_bottom = plt.subplot(2, len(classes_combinations), i+len(classes_combinations)+1)
            tpr, fpr = self.get_all_roc_coordinates(df_aux['class'], df_aux['prob'],prob_bins)
            self.plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
            ax_bottom.set_title("ROC Curve %dv%d"%(c1,c2))

            # Calculates the ROC AUC OvO
            roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])

        # Displays the ROC AUC for each class
        avg_roc_auc = 0
        i = 0
        for k in roc_auc_ovo:
            avg_roc_auc += roc_auc_ovo[k]
            i += 1
            print(f"{k} ROC AUC OvO: {roc_auc_ovo[k]:.4f}")
        print(f"average ROC AUC OvO: {avg_roc_auc/i:.4f}")
        plt.tight_layout()
        plt.show()

def main():
    df_sample = CSHSample.get_random_classification(1000,n_feature=10,n_class=2)
    df_train,df_test = CSHSample.split_dataset(df_sample)

if __name__ == "__main__":
    main()
