#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env bash
import matplotlib.pyplot as plt
import seaborn as sns
from pyecharts import options as opts
from pyecharts.charts import Sankey
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import subprocess

class CSHCommon:
    def __init__(self):
        pass
    '''
    带进度条，读取csv文件
    '''
    @staticmethod
    def load_csv(file_name,chunksize=1000):
        out = subprocess.getoutput("wc -l %s" % file_name)
        total = int(out.split()[0]) / chunksize
        return pd.concat([chunk for chunk in tqdm(pd.read_csv(file_name, chunksize=chunksize),total=total, desc='Loading data %s'%file_name)])
    
    '''
    找到序列中，发生变化的index
    '''
    @staticmethod
    def find_changed_index(ds):
        df = pd.DataFrame()
        ds_1 = ds.fillna(0.00000000001)
        ds_1 = ds_1.append(pd.Series([ds.iloc[-1]]))
        ds_2 = ds_1.shift(1)
        df['x'] = ds_1
        df['y'] = ds_2
        df=df[0:-1]
        return df[df['x']!=df['y']].index.values

    '''
    使用PCA，得到降维后和重构后的矩阵（根据特征值和特征向量）
    dataMat：原始矩阵
    n：特征向量的维度（降维后）
    '''
    @staticmethod
    def get_matrix_by_pca(dataMat, n):
        # 零均值化
        def zeroMean(dataMat):
            meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
            newData = dataMat - meanVal
            return newData, meanVal

        newData, meanVal = zeroMean(dataMat)
        covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        # argsort将x中的元素从小到大排列，提取其对应的index(索引)
        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
        # print(eigValIndice)
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
        lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
        return lowDDataMat, reconMat

def main():
     CSHCommon.load_csv("../../data/profile.csv")

#    ds = pd.Series([1,1,1,1,2,3,3,3,3,3,5])
#    print("Changed node index ",CSHCommon.find_changed_index(ds))
    
if __name__ == "__main__":
    main()



