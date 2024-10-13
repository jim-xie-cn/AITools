'''
数据探索
1. CSHDataDistribution (数据分布)
    --显示直方图、KED图和Q-Q图
    --检查是否为高斯分布
    --比较两个分布是否相私
2. CSHDataTest（假设检验）
    --检验是否为平稳序列
    --显示自相关和偏自相关图
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
from scipy.stats import norm, expon
from scipy.optimize import curve_fit

class CSHDataDistribution:
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
        ret['kolmogorov-smirnov']['describe'] = "Kolmogorov-Smirnov检验是一项拟合优度的统计检验。 此测试比较两个分布（在这种情况下，两个分布之一是高斯分布）。 此检验的零假设是，两分布相同（或），两个分布之间没有差异。"
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
    可视化分布
    '''
    @staticmethod
    def normal_fit(ds_data,bins=100):
        data = ds_data.to_numpy()
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        
        def normal_pdf(x, mu, sigma):
            return norm.pdf(x, loc=mu, scale=sigma)
        def exponential_pdf(x, lamb):
            return expon.pdf(x, scale=1/lamb)
        
        params_normal, _ = curve_fit(normal_pdf, bin_centers, hist)
        params_exponential, _ = curve_fit(exponential_pdf, bin_centers, hist)
        
        plt.hist(data, bins=bins, density=True, alpha=0.5, color='blue')
        
        x_range = np.linspace(-4, 4, 1000)
        plt.plot(x_range, normal_pdf(x_range, *params_normal), color='red', label='Normal Fit')
        plt.plot(x_range, exponential_pdf(x_range, *params_exponential), color='green', label='Exponential Fit')
        
        plt.title('Fitting Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        
class CSHDataTest:

    '''
    稳定性测试，平稳序列可以使用ARIMA
    1.判断是否为白噪声，白噪声满足ARIMA
        -- 平稳、独立和等方差
        -- 自相关和偏自相关函数在所有滞后阶数上接近于零
        -- 白噪声适合使用ARIMA模型进行预测
    2. 满足以下条件，可以使用ARIMA
        #ADF Test result同时小于1%、5%、10%
        #P-value (不变显著性) 接近0。
    '''
    @staticmethod
    def stationarity_test(timeSeries):
        dftest = adfuller(timeSeries)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        del dfoutput['#Lags Used']
        del dfoutput['Number of Observations Used']
        return dfoutput

    '''
    绘制自相关ACF(与不同滞后阶数之间的相关性),和偏自相关性PACF（与特定滞后阶数的相关性）
    1. 自相关系数ACF：x轴表示滞后阶数（lag），y轴表示自相关系数
       -- 自相关性：自相关系数在某个滞后阶数上远离0，说明在这个滞后阶数上存在显著的自相关
       -- 显著性区域：自相关系数超过置信区间的边界，则认为该系数是显著的
       -- 截尾特性：随着滞后阶数的增加，自相关系数逐渐减小并趋于零，这种情况下，时间序列可能是平稳的，可以使用ARMA模型进行建模
       -- 周期性： 在某些滞后阶数上显示出周期性或周期性衰减。
    2. PACF
       -- 偏自相关性：x轴表示滞后阶数（lag），y轴表示偏自相关系数
       -- 显著性区域：如果偏自相关系数超过置信区间的边界，则认为该系数是显著的
       -- 截尾特性： 随着滞后阶数的增加，偏自相关系数逐渐减小并趋于零。这种情况下，时间序列可能是平稳的，可以使用AR模型进行建模
       -- 选择模型阶数：在某个滞后阶数上截尾，而在后续的滞后阶数上迅速衰减至接近零，那么该滞后阶数可能是适合的AR模型的阶数。
    '''
    @staticmethod
    def show_acf_pacf(timeSeries):
        plot_acf(timeSeries).show()
        plot_pacf(timeSeries).show()

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
    #生成测试数据
    np.random.seed(11)
    X_Normal = np.random.randn(1000)
    X_linear = np.linspace(1,100,1000)
    df_data = pd.DataFrame()
    df_data['Normal'] = pd.Series(X_Normal)
    df_data['Linear'] = pd.Series(X_linear)

    #是否为高斯分布
    test = CSHDataDistribution.normal_test(df_data['Normal'])
    print(json.dumps(test,indent=4))
    test = CSHDataDistribution.dist_test(df_data['Normal'],df_data['Normal'])
    print(json.dumps(test,indent=4))
    CSHDataDistribution.show_dist(df_data['Normal'],bins = 20)

    #平稳性检查
    df_test = CSHDataTest.stationarity_test(df_data.Normal)
    print(df_test)
    CSHDataTest.show_acf_pacf(df_data.Normal)
    plt.show()

if __name__ == "__main__":
    main()
