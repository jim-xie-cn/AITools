import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
'''
数据预处理类
'''
class CSHDataProcess:
    def __init__(self):
        pass
    '''
    将数据转换为正态分布（使用box-cox）
    lambda_value==0，使用log转换，lambda_value==None，自动计算最近lambda参数
    '''
    @staticmethod
    def normal_transform(ds_data,lambda_value=None):
        t_data, l_value = stats.boxcox(ds_data,lambda_value)
        return pd.Series(t_data),l_value
    '''
    从正态分布中，将数据还原（使用box-cox）
    '''
    @staticmethod
    def normal_recovery(ds_data,lambda_value):
        original_data = inv_boxcox(ds_data,lambda_value)
        return pd.Series(original_data)
    '''
    获取异常数据（Z-Score绝对值大于3 sigma）
    ''' 
    @staticmethod
    def get_abnormal(ds_data,n_sigma=3):
        ds_z_score = (ds_data - ds_data.mean()) / ds_data.std()
        filtered =  ds_z_score[(ds_z_score<(0-n_sigma) ) | (ds_z_score>n_sigma)]
        return ds_data.iloc[filtered.index]

    '''
    稳定性测试，平稳序列可以使用ARIMA
    1.判断是否为白噪声，白噪声满足
        -- 平稳、独立和等方差
        -- 自相关和偏自相关函数在所有滞后阶数上接近于零
        -- 白噪声不适合使用ARIMA模型进行预测
    2. 满足以下条件，可以使用ARIMA
        #ADF Test result同时小于1%、5%、10%
        #P-value (不变显著性) 接近0。
    '''
    @staticmethod
    def check_stationarity(timeSeries):
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

def main():
    ds_data = pd.Series([1,2,3,4,5,6])
    transformed_data = CSHDataProcess.get_abnormal(ds_data)
    ds_data = ds_data.drop(transformed_data.index)
    #ds_data = ds_data[ds_data>0]
    ds_data.plot()

    CSHDataProcess.show_acf_pacf(ds_data)
    CSHDataProcess.check_stationarity(ds_data)
    plt.show()    

if __name__ == "__main__":         
    main()                         
