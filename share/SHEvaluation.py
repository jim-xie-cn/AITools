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
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore") 
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)
# 使用文泉驿字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

class CSHEvaluate():
    
    def __init__(self):
        pass
    
    @staticmethod
    def evaluate(ds_pred,ds_true,average='weighted'):
        result = []
        y_pred = ds_pred.tolist()
        y_true = ds_true.tolist()
        tmp = {}
        tmp['confusion_matrix'] = confusion_matrix(y_true,y_pred)
        tmp['recall'] = recall_score(y_true,y_pred,average=average)
        tmp['mcc'] = matthews_corrcoef(y_true,y_pred)
        tmp['accuracy'] = accuracy_score(y_true,y_pred)
        tmp['precision'] = precision_score(y_true,y_pred,average=average)
        if ds_true.nunique() == 1:
            tmp['auc'] = "error true: 1 class"
            tmp['f1_score'] = "error True: 1 class"
            tmp['fbeta_score'] = "error True: 1 class"
        elif ds_pred.nunique() == 1:
            tmp['auc'] = "error pred: 1 class"
            tmp['f1_score'] = "error pred: 1 class"
            tmp['fbeta_score'] = "error pred: 1 class"
        elif ds_true.nunique() > 2:
            tmp['auc_ovo'] = roc_auc_score(y_true,y_prab,multi_class='ovo',average=average)
            tmp['auc_ovr'] = roc_auc_score(y_true,y_prab,multi_class='ovr',average=average)
        else:
            tmp['auc'] = roc_auc_score(y_true,y_pred,average=average)
            tmp['f1_score'] = f1_score(y_true,y_pred,average=average)
            tmp['fbeta_score'] = fbeta_score(y_true,y_pred,beta=0.5,average=average)
        result.append(tmp)
        return pd.DataFrame(result)
        
    @staticmethod
    def get_binary_roc(ds_true,ds_prob):
        fpr, tpr, thresholds = roc_curve(ds_true, ds_prob)
        roc_auc = auc(fpr, tpr)
        df_roc = pd.DataFrame()
        df_roc['fpr'] = fpr
        df_roc['tpr'] = tpr
        df_roc['thresholds'] = thresholds
        return roc_auc,df_roc
    
    @staticmethod
    def show_binary_roc(roc_auc,df_roc,title = None):
        plt.figure()
        plt.plot(df_roc['fpr'], df_roc['tpr'], color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 对角线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if title != None:
            plt.title('ROC of %s'%title)
        else:
            plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

def main():
    df_sample = CSHSample.get_random_classification(1000,n_feature=10,n_class=2)
    df_train,df_test = CSHSample.split_dataset(df_sample)

if __name__ == "__main__":
    main()
