import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

'''
封装常见的Tensor操作
'''
class CPTTensor():
    '''
    将numpy数组转换为Tensor
    x：numpy数组
    isGPU: 是否使用GPU，默认不使用
    返回值：FloatTensor
    '''
    @staticmethod    
    def load_numpy(x,isGPU = False):
        if isGPU:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        return torch.from_numpy(x).type(dtype)
    '''
    将Tensor转换为numpy数组
    x_tensor：Tensor
    返回值：numpy数组
    '''
    @staticmethod
    def to_numpy(x_tensor):
        return x_tensor.cpu().detach().numpy()

def main():
    x1 = np.random.normal(loc=0, scale=5, size=1000)
    x2 = np.random.normal(loc=0, scale=5, size=1000)
    x = np.dstack((x1,x2)).squeeze()
    y = (x[:,0]+x[:,1]).reshape(-1,1)
    x = CPTTensor.load_numpy(x)
    y = CPTTensor.load_numpy(y)
    x , y =(Variable(x,requires_grad=False),Variable(y,requires_grad=False))
    print(x.shape,y.shape)

if __name__ == '__main__':
    main()
