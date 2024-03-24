'''
处理数据分布
1. 检测是否为高斯分布
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import chisquare
import seaborn as sns
import json
from scipy.stats import f_oneway
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests

class CSHDistribution:
    def __init__(self):
        pass
    '''
    显示分布图
    '''
    @staticmethod
    def show_dist(ds,bins = 10 ):
        fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(8, 3))
        ds.hist(ax=ax[0],bins=bins)
        ax[0].set_title("Histogram")
        ds.plot(kind="kde",ax=ax[1])
        ax[1].set_title("KDE")
        probplot(x=ds,dist='norm',plot=ax[2])
        ax[2].set_title("Q-Q")
        plt.show()
    
    '''
    检查是否为高斯分布
    '''
    @staticmethod
    def normal_test(ds , p_value = 0.05):
        ret = {}
        ret['shapiro-wilk'] = {}
        temp = shapiro(ds)
        ret['shapiro-wilk']['is_normal'] = str(temp[1] >= p_value)
        ret['shapiro-wilk']['detail'] = temp
        ret['shapiro-wilk']['describe'] = "Shapiro-Wilk检验，适用于小样本场合（3≤n≤50），受异常值影响较大。"

        ret['kolmogorov-smirnov'] = {}
        temp = kstest(ds,"norm")
        ret['kolmogorov-smirnov']['is_normal'] = str(temp[1] >= p_value)
        ret['kolmogorov-smirnov']['detail'] = temp
        ret['kolmogorov-smirnov']['describe'] = "Kolmogorov-Smirnov检验是一项拟合优度的统计检验。 此测试比较两个分布（在这种情况下，两个分布之一是高斯分布）。 此检验的零假设是，两个分布相同（或），两个分布之间没有差异。"

        ret['skewness-kurtosis'] = {}
        temp = normaltest(ds,)
        ret['skewness-kurtosis']['is_normal'] = str(temp[1] >= p_value)
        ret['skewness-kurtosis']['detail'] = temp
        ret['skewness-kurtosis']['describe'] = "DAgostino-Pearson方法使用偏度和峰度测试正态性。 该检验的零假设是，分布是从正态分布中得出的。"

        return ret
    
    '''
    检查两个分布是否相同
    '''
    @staticmethod
    def dist_test(ds1, ds2 , p_value = 0.05):
        ret = {}
        ret['f-test'] = {}
        temp = f_oneway(ds1,ds2)
        ret['f-test']['is_same'] = str(temp[1] >= p_value)
        ret['f-test']['detail'] = temp
        ret['f-test']['describe'] = "F检验(方差分析)。"
        
        ret['chis-test'] = {}
        temp = chisquare(f_obs=ds1,f_exp=ds2)
        ret['chis-test']['is_same'] = str(temp[1] >= p_value)
        ret['chis-test']['detail'] = temp
        ret['chis-test']['describe'] = "卡方检验。"
        return ret

    '''
    检查是否为平稳序列
    '''
    @staticmethod
    def stationary_test(ds,show_graph=True, p_value = 0.05):
        ret = {}
        ret['adf-test'] =  {}
        temp = adfuller(ds , autolag = 'AIC')  # ADF检验
        t = temp[0]
        p = temp[1]
        used_lag = temp[2]
        ret['adf-test']['is_stationary'] = str(p <= p_value )
        ret['adf-test']['detail'] = temp
        ret['adf-test']['describe'] = "单位根检测。"
        
        if show_graph:
            fig, ax =plt.subplots(1,2,constrained_layout=True, figsize=(8, 2))
            plot_acf( ds , ax = ax[0] )
            plot_pacf( ds ,ax = ax[1] )
            plt.show()

        return ret
    
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
    np.random.seed(11)
    X_Normal = np.random.randn(1000)
    X_linear = np.linspace(1,100,1000)
    df_data = pd.DataFrame()
    df_data['Normal'] = pd.Series(X_Normal)
    df_data['Linear'] = pd.Series(X_linear)
    test = CSHDistribution.normal_test(df_data['Normal'])
    print(json.dumps(test,indent=4))
    test = CSHDistribution.dist_test(df_data['Normal'],df_data['Normal'])
    print(json.dumps(test,indent=4))
    CSHDistribution.show_dist(df_data['Normal'],bins = 20)
    CSHDistribution.stationary_test(df_data.Normal)

    a = [1,-1,2,-2,3,-3.1]
    b = [2,-2,3,-3,4,-4.1]
    result = CSHDistribution.granger_test(b,a,[1])
    print(result)

    a.extend(a)
    b.extend(b)
    result = CSHDistribution.granger_test(b,a,2)
    print(result)

if __name__ == "__main__":
    main()
