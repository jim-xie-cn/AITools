'''
数据预处理
1. CSHDataProcess (数据预处理)
   --将数据转换为正态分布（box-cox）
   --从正态分布中，将数据还原（box-cox）
   --获取异常数据（Z-Score）
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from SHSample import CSHSample
from sklearn.preprocessing import StandardScaler

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
        return pd.Series(t_data,index=ds_data.index),l_value
    '''
    从正态分布中，将数据还原（使用box-cox）
    '''
    @staticmethod
    def normal_recovery(ds_data,lambda_value):
        original_data = inv_boxcox(np.array(ds_data.values),lambda_value)
        return pd.Series(original_data,index=ds_data.index)
    '''
    多项式拟合
    ds_x：pd.Series,自变量，当为空时，时间序列数据拟合。
    ds_y：pd.Series,因变量
    degree：多项式最高阶数
    返回值：拟合对象(p_object),拟合残差的标准差(e_std),拟合后的数据(v_object)
    '''
    def polyfit(ds_y,ds_x = pd.Series([]), degree = 2):
        if ds_x.empty :
            x = np.arange(ds_y.shape[0]) # 生成对应的时间点作为（自变量）
        else:
            x = ds_x.values()
        y = ds_y.values
    
        p_coefficients, e_residuals, _, _, _  = np.polyfit(x, y, degree,full=True)
    
        p_object = np.poly1d(p_coefficients)
        v_object = pd.Series(p_object(x),index=ds_y.index)
        e_std = np.sqrt(e_residuals / len(x))
        return p_object,e_std,v_object
    '''
    获取异常数据（Z-Score绝对值大于3 sigma）
    ''' 
    @staticmethod
    def get_abnormal(ds_data,n_sigma=3):
        ds_z_score = (ds_data - ds_data.mean()) / ds_data.std()
        filtered =  ds_z_score[(ds_z_score<(0-n_sigma) ) | (ds_z_score>n_sigma)]
        return ds_data.iloc[filtered.index]

    # 对特征字段进行标准化处理(z-score/max-min)
    @staticmethod
    def get_scale(df_data,x_columns=[],y_column='label',scale_type='z-score'):

        df_ret = df_data.copy(deep=True)
        if x_columns:
            scale_colums = x_columns
        else:
            scale_colums = []
            for key,type in zip(df_data.keys(),df_data.dtypes):
                if not type in ["bool","object","category"] and not key in [y_column]:
                    scale_colums.append(key)

        if scale_type == 'z-score':
            scaler = StandardScaler()
            df_ret[scale_colums] = scaler.fit_transform(df_ret[scale_colums])
        elif scale_type == 'max-min':
            for key in scale_colums:
                df_ret[key] = (df_ret[key]-df_ret[key].min())/(df_ret[key].max()-df_ret[key].min())

        return df_ret,scale_colums

def main():
    ds_data = pd.Series([1,2,3,4,5,6])
    transformed_data = CSHDataProcess.get_abnormal(ds_data)
    ds_data = ds_data.drop(transformed_data.index)
    #ds_data = ds_data[ds_data>0]
    ds_data.plot()
    plt.show()    
    
    p_object,e_std,v_object = CSHDataProcess.polyfit(ds_y=ds_data,degree=3)
    print(v_object)
    
    df_sample = CSHSample.get_random_classification(1000,n_feature=10,n_class=2)
    df_scale,scale_colums = CSHDataProcess.get_scale(df_sample,y_column='y',scale_type="max-min")
    print(df_scale)

if __name__ == "__main__":         
    main()                         
