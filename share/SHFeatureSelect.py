'''
特征选择
1. CSHFeature
   --生成信息图(分类数据)
   --GLM-ANOVA 方差检验（回归数据）
   --格兰特因果检验（时序数据）
'''
import numpy as np
import pandas as pd
import h2o,json,os,sys,datetime,warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from SHCommon import CSHCommon
from SHSample import CSHSample
from tqdm.notebook import tqdm
from h2o.automl import H2OAutoML
from h2o.estimators import *
from warnings import filterwarnings
from statsmodels.tsa.stattools import grangercausalitytests
filterwarnings("ignore") 
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)

class CSHFeature():
 
    def __del__(self):
       pass
       #h2o.shutdown()

    def __init__(self,ip='localhost',port=54321):
        #h2o.init(nthreads = -1, verbose=False)
        self.m_df_h2o = None
        self.m_col_x = []
        self.m_col_y = None

    def load(self,df_sample,x_columns = [],y_column='label',is_regression = False):
        
        if x_columns == []:
            total_columns = df_sample.keys().tolist()
        else:
            total_columns = x_columns
         
        if y_column in total_columns:
            total_columns.remove(y_column)
                
        self.m_col_x = total_columns
        self.m_col_y = y_column
        
        self.m_df_h2o = h2o.H2OFrame(df_sample)
        if not is_regression:
            self.m_df_h2o[self.m_col_y] = self.m_df_h2o[self.m_col_y].asfactor()
    '''
    横坐标：信息总量（total information）
        --变量对预测的影响，,即该变量与其他变量的相关性。
        --横轴上的值越大，表示变量对响应的影响越显著。
    纵坐标：净信息（net information）
        --变量的独特性，总信息量.
        --净信息越高，预测能力越强，表示该变量对响应的影响越独特。
    可接受特征
        --位于虚线以上和右侧的特征是最具预测能力和独特性的特征，它们被认为是可接受的特征，
        --这些特征是被认为是核心驱动因素的变量，它们在总信息（预测能力）和净信息（独特性）方面都表现出色。        
        --可以用于建立模型和做出决策。
    返回值方法：
    ig.get_admissible_score_frame()
    ig.get_admissible_features()
    '''
    def get_inform_graph(self,algorithm="AUTO",protected_columns=[]): #["All",'AUTO','deeplearning','drf','gbm','glm','xgboost']
        if algorithm == "All":
            ret = {}
            for algor in ['AUTO','deeplearning','drf','gbm','glm']: #,'xgboost']: xgboost 有 bug
                if protected_columns:
                    ig = H2OInfogram(algorithm=algor,protected_columns=protected_columns)
                else:
                    ig = H2OInfogram(algorithm=algor)
                ig.train(x=self.m_col_x, y=self.m_col_y,training_frame=self.m_df_h2o)
                ret[algor] = ig
            return ret
        else:
            if protected_columns:
                ig = H2OInfogram(algorithm=algorithm,protected_columns=protected_columns)
            else:
                ig = H2OInfogram(algorithm=algorithm)
            ig.train(x=self.m_col_x, y=self.m_col_y,training_frame=self.m_df_h2o)
            return ig
    '''
    统计自变量和因变量的相关性
        --p值小于0.05,认为有相关性
        --通过特征组集合，查看特征之间是否有相关性
    '''           
    def get_anovaglm(self,family='gaussian',lambda_ = 0,highest_interaction_term=2):
        anova_model = H2OANOVAGLMEstimator(family=family,
                                   lambda_=lambda_,
                                   missing_values_handling="skip",
                                   highest_interaction_term=highest_interaction_term)
        anova_model.train(x=self.m_col_x, y=self.m_col_y, training_frame=self.m_df_h2o)
        return anova_model
        #anova_model.summary()

    '''
    格兰特因果检验
    ds_result：结果序列（必须为平稳序列）
    ds_source：原因序列（必须为平稳序列）
    maxlag：时间间隔（整数或列表，为整数时，遍历所有的lag）
    返回值：
    1. 最小的p值
    2. 最佳lag
    3. approve_list通过测试的lag
    4. 详细测试结果
    '''
    @staticmethod
    def granger_test(ds_result,ds_source,maxlag,p_value = 0.05):

        df_test = pd.DataFrame()
        df_test['result'] = ds_result
        df_test['source'] = ds_source

        gc_result = grangercausalitytests(df_test[['result', 'source']], maxlag=maxlag,verbose=False )
        min_lag_p = 100
        best_lag = -1
        detail = {}
        approve_list = []
        for lag in gc_result:

            result = gc_result[lag][0]
            detail[lag] = result.copy()
    
            max_p = 0 
            for test in result:
                test_value = result[test][0]
                test_p = result[test][1]
                if test_p > max_p:
                    max_p = test_p

            if max_p > p_value:
                continue
            #if max_p == 0:
            #    continue   
            approve_list.append({'lag':lag,"max p-value":max_p})
            
            if min_lag_p > max_p:
                min_lag_p = max_p
                best_lag = lag 

        return min_lag_p, best_lag, approve_list, detail   

def main():
    h2o.init(nthreads = -1, verbose=False)

    #分类数据,查看信息图
    df_sample = CSHSample.get_random_classification(1000,n_feature=10,n_class=2)
    Fea = CSHFeature()
    Fea.load(df_sample,x_columns = ['x1','x2'],y_column='y',is_regression = False)
    ig = Fea.get_inform_graph("AUTO")
    ig.plot()
    ig.show()
    #回归数据，查看方差检验
    df_sample = CSHSample.get_random_regression()
    Fea.load(df_sample,x_columns = ['x1','x2'],y_column='y',is_regression = True)
    ag = Fea.get_anovaglm()
    print(ag.summary())

    #格兰特因果检验
    a = [1,-1,2,-2,3,-3.1]
    b = [2,-2,3,-3,4,-4.1]
    result = CSHFeature.granger_test(b,a,[1])
    print(result)
        
    a.extend(a)
    b.extend(b)
    result = CSHFeature.granger_test(b,a,2)
    print(result)
if __name__ == "__main__":
    main()
