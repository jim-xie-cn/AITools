'''
定义深度学习模型
'''
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os,sys,json
from PTTensor import CPTTensor
from PTSample import CPTSample

'''
定制化神经元计算方法（默认是加权求和）
'''
class CPTMethod(torch.nn.Module):
    '''
    初始化计算方法对象
    in_features : 输入的特征数（列）
    out_features: 神经元数量
    '''
    def __init__(self,in_features,out_features):
        super().__init__()
        self.m_weight = nn.Parameter(torch.randn(out_features,in_features))
        self.m_bias = nn.Parameter(torch.randn(out_features))
    '''
    计算方法实现
    '''
    def forward(self, x):
        return torch.mm(x, self.m_weight.t() ) + self.m_bias
    '''
    使用范例
    '''
    @staticmethod
    def test():
        x,y = CPTSample.get_sample()
        print("x:",x.shape)
        print("in_features:",2)
        print("out_features:",3)
        module = CFOCMethod(2,3)
        print("weight:",module.m_weight.shape)
        print("bias:",module.m_bias.shape)
        out = module.forward(x)
        print("out:",out.shape)
'''
定制激活函数（默认是relu）
'''
class CPTActivation(nn.Module):
    '''
    初始化激活函数
    '''
    def __init__(self, params = {}):
        super().__init__()
        self.params = params
    '''
    激活函数实现
    '''
    def forward(self, x):
        return F.relu(x)
    '''
    使用范例
    '''
    @staticmethod
    def test():
        x,y = CPTSample.get_sample()
        print("y:",y.shape)
        module = CPTActivation()
        out = module.forward(y)
        print("out:",out.shape)
'''
定义深度学习模型
1. 定义网络的结构（包括权重层和激活层）
2. 设置损失函数（默认是MSE）
3. 设置优化算法（默认是SGD）
'''
class CPTModel():
    '''
    初始化模型对象
    '''
    def __init__(self):
        self.m_model = None
    '''
    数据前向移动
    '''
    def forward(self, in_data):
        return self.m_model.forward(in_data)
    '''
    获取计算图
    '''
    def get_graph(self,x):
        return make_dot(self.m_model(x), params=dict(self.m_model.named_parameters()))
    '''
    返回pytorch模型对象
    '''
    def get_model(self):
        return self.m_model
    '''
    获取网络结构
    '''
    def get_layers(self):
        return self.get_model().named_children()
    '''
    获取各个层的输出
    '''
    def get_output(self,x):
        result = []
        data = x
        for item in self.get_layers():
            tmp = {}
            tmp['layer_id'] = len(result)
            tmp['layer_name'] = item[0]
            tmp['input'] = data
            data = item[1](data)
            tmp['output'] = data
            result.append(tmp)
        return result
        
    '''
    设置损失函数
    '''
    def set_loss(self):
        self.m_loss = torch.nn.MSELoss()
    '''
    设置优化算法
    '''
    def set_optimize(self):
        self.m_optimizer=torch.optim.SGD(self.m_model.parameters(), lr=1e-4)
    '''
    构建网络
    '''
    def build(self):
        self.m_model = nn.Sequential()
        self.m_model.add_module('W0', nn.Linear(2,5))
        self.m_model.add_module('A1', CFOCActivation())
        self.m_model.add_module('W1', CFOCMethod(5,1))
        self.m_model.add_module('A2', CFOCActivation())
        
        self.set_loss()
        self.set_optimize()
    '''
    有标签训练
    '''
    def train(self,x,y):
        y_pred = self.m_model(x)
        loss = self.m_loss(y_pred, y)
        self.m_optimizer.zero_grad()
        loss.backward()
        self.m_optimizer.step()
        return loss,y_pred
    '''
    预测
    '''
    def predict(self,x):
        return self.m_model(x)

'''
测试深度学习模型
'''
class CTestModel(CPTModel):

    def __init__(self):
        super().__init__()
    
    def set_loss(self):
        self.m_loss = torch.nn.MSELoss()
    
    def set_optimize(self):
        self.m_optimizer=torch.optim.SGD(self.m_model.parameters(), lr=1e-4)
        
    def build(self):
        self.m_model = nn.Sequential()
        self.m_model.add_module('W0', nn.Linear(2,5))
        self.m_model.add_module('A1', CPTActivation())
        self.m_model.add_module('W1', CPTMethod(5,1))
        self.m_model.add_module('A2', CPTActivation())
        
        self.set_loss()
        self.set_optimize()

def main():
    x,y = CPTSample.get_sample()
    model = CTestModel()
    model.build()
    for i in range(1000):
        loss,y_pred = model.train(x,y)
        if i % 50 == 0:
            print(i,loss.item())

    print("archtechture")
    print(model.m_model)

    all_output = model.get_output(x)
    print("hidden output")
    print(all_output)

    x_test,y_test = CPTSample.get_sample(3)
    y_pred = model.predict(x_test)
    print("predict test")
    print(x_test)
    print(y_test)
    print(y_pred)

if __name__ == '__main__':
    main()
