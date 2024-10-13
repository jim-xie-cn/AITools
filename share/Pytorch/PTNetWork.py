'''
使用pytorch创建神经网络
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from PTTensor import CPTTensor
import json

'''
封装常见的pytorch模型操作
'''
class CPTNetBase(nn.Module):
    '''
    初始化Module
    m_layer:每层网络的定义
    m_loss:损失函数
    m_optimizer:优化器
    '''
    def __init__(self):
        super(CPTNetBase,self).__init__()
        self.m_layer = []
        self.m_loss = None
        self.m_optimizer = None
    
    def Save(self,file_name):
        torch.save(self, file_name)

    @staticmethod
    def Load(file_name):
        return torch.load(file_name)

    '''
    获取网络的参数
    返回值：可训练参数列表迭代器
    '''
    def get_parameters(self):
        params = nn.ParameterList()
        for item in self.m_layer:
            params = params.extend(item['param'])
        return params
    '''
    进行一次数据正向传输
    in_data：输入样本
    返回值：网络输出（如果定义激活函数，则输出激活后的值）
    '''
    def forward(self, in_data):
        out = in_data
        for item in self.m_layer:
            out = item['method'](out)
            if item['active']:
                out = item['active'](out)
        return out
    '''
    获取模型结构
    返回值：每层网络的定义
    '''
    def Get_Model(self):
        return self.m_layer
    '''
    获取模型信息
    返回值：模型信息，包括每层的权重、偏置、激活函数和训练的参数
    '''
    def Get_Info(self):
        info = {}
        for layer in self.Get_Model():
            name = layer['name']
            info[name] = {}
            info[name]['method'] = str(layer['method'])
            info[name]['active'] = str(layer['active'])
            info[name]['param'] = []
            p = layer['param']
            tmp = {}
            tmp['weight'] = CPTTensor.to_numpy(p[0].data).tolist()
            tmp['bias'] = CPTTensor.to_numpy(p[1].data).tolist()
            info[name]['param'].append(tmp)
        return info
    '''
    增加一个隐含或输出层
    method：神经元计算方法（一般为加权求和）
    active：神经元的激活函数
    params：待训练的参数
    name：这一层的命名
    '''
    def Add_Layer(self,method,active,params,name=''):
        layer = {}
        if not name:
            layer['name'] = "L_%d"%len(self.m_layer)
        else:
            layer['name'] = name
        layer['method'] = method
        layer['active'] = active
        layer['param'] = list(params)

        self.m_layer.append(layer)
    '''
    设置损失函数
    loss：损失函数
    '''    
    def Set_Loss(self,loss):
        self.m_loss = loss
    '''
    设置优化算法
    optimizer：优化算法
    **kw：算法对应的参数
    '''     
    def Set_Optimizer(self,optimizer,**kw):
        params = nn.ParameterList()
        for item in self.m_layer:
            params = params.extend(item['param'])
        self.m_optimizer = optimizer(params,**kw)
    '''
    进行一次训练，包含以下4步：
        1.数据正向传输
        2.计算损耗值
        3.误差反传
        4.权重更新
    返回值：损失值和预测结果
    '''     
    def Train(self,x, y):
        output = self(x)
        loss = self.m_loss(output,y)
        self.m_optimizer.zero_grad()
        loss.backward()
        self.m_optimizer.step()
        return loss, output
    '''
    进行一次预测
    in_data:输入值
    返回值：预测结果
    '''
    def Predict(self,in_data):
        return self(in_data)

'''
定义神经网络，包含以下几步：
1.设置神经元计算定义（set_method）
2.设置损失函数定义（set_loss）
3.设置优化算法定义（set_optimizer）
4.设置网络结构（set_network）
5.生成网络（build）
'''
class CPTNet(CPTNetBase):
    '''
    初始化Module
    '''
    def __init__(self):
        super(CPTNet,self).__init__()
    '''
    设置神经元计算函数
    in_dim：输入向量维度
    out_dim：输出向量维度
    返回值一：计算定义函数
    返回值二：可训练参数
    '''
    def set_method(self,in_dim,out_dim):
        method = nn.Linear(in_dim,out_dim)
        params = method.parameters()
        return method,params
    '''
    设置损失函数
    '''    
    def set_loss(self):
        loss = torch.nn.MSELoss()
        self.Set_Loss(loss)
    '''
    设置优化算法
    '''
    def set_optimizer(self):
        optimizer = torch.optim.SGD
        self.Set_Optimizer(optimizer,lr=1e-4, momentum=0.9)    
    '''
    设置网络结构
    n_input：输入维度
    n_output：输出维度
    '''
    def set_network(self,n_input,n_output):
        #增加第一个隐含层
        n_hidden_1 = 2
        active_1 = F.relu
        method ,params = self.set_method(n_input,n_hidden_1)
        self.Add_Layer(method,active_1,params,'hdden_1')
    
        #增加第2个隐含层
        n_hidden_2 = 3
        active_2 = F.relu
        method ,params = self.set_method(n_hidden_1,n_hidden_2)
        self.Add_Layer(method,active_2,params,'hdden_2')
        
        #增加输出层
        active = None # None is for regression
        method ,params = self.set_method(n_hidden_2,n_output)
        self.Add_Layer(method,active,params,'output')
    '''
    生成网络
    n_input：输入维度
    n_output：输出维度
    '''
    def build(self,n_input,n_output):
        self.set_network(n_input,n_output)
        self.set_loss()
        self.set_optimizer()

'''
自定义神经网络
'''
class CTestNet(CPTNet):
    def __init__(self):
        super(CTestNet,self).__init__()
    def set_method(self,in_dim,out_dim):
        return super().set_method(in_dim,out_dim)
    def set_loss(self):
        super().set_loss()
    def set_optimizer(self):
        super().set_optimizer()
    def set_network(self,n_input,n_output):
        super().set_network(n_input,n_output)
    def build(self,n_input,n_output):
        super().build(n_input,n_output)

def main():
    #prepare dataset
    x1 = np.random.normal(loc=0, scale=5, size=1000)
    x2 = np.random.normal(loc=0, scale=5, size=1000)
    x = np.dstack((x1,x2)).squeeze()
    y = (x[:,0]+x[:,1]).reshape(-1,1)
    x = CPTTensor.load_numpy(x)
    y = CPTTensor.load_numpy(y)
    x , y =(Variable(x,requires_grad=False),Variable(y,requires_grad=False))
    print(x.shape,y.shape)
    
    #build module
    model = CTestNet()
    model.build(2,1)

    info = model.Get_Info()
    print("model info")
    print(json.dumps(info,indent=4))
    for i in range(500):
        loss,output = model.Train(x,y)
        if i % 50 ==0:
            print("trained %d,loss:%f"%(i,loss.data))
    print("model info")
    info = model.Get_Info()
    print(json.dumps(info,indent=4))

    print("predict")
    
    x_test =torch.Tensor([1.2,2.2])
    y_pred = model.Predict(x_test)
    print("input",x_test.data,"output",y_pred.data)

if __name__ == '__main__':
    main()
