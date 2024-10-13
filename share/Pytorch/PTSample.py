'''
数据集处理
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from PTTensor import CPTTensor
import json
import matplotlib.pyplot as plt
import seaborn as sns

class CPTSample():
    @staticmethod 
    def get_sample(n_sample = 10 ):
        x = np.random.rand(n_sample,2)
        y = []
        for item in x:
            t = item[0] + item[1]
            y.append(np.sin(t))
        y = np.array(y).reshape(-1,1)
        x = CPTTensor.load_numpy(x)
        y = CPTTensor.load_numpy(y)
        x , y =(Variable(x,requires_grad=False),Variable(y,requires_grad=False))
        return x,y
    
    @staticmethod 
    def show_sample():
        sns.lineplot(x=(x[:,0]+x[:,1]),y=y.squeeze())
        plt.show()

def main():
    x,y = CPTSample.get_sample()
    print("x",x.shape)
    print("y",y.shape)

if __name__ == '__main__':
    main()
