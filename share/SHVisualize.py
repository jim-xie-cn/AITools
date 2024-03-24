#!/usr/bin/env bash
#!-*- coding:utf8-*-
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pyecharts import options as opts
from pyecharts.charts import Sankey
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from IPython.core.display import display, HTML
from matplotlib.ticker import FuncFormatter
def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.2E' % x
set_scientific = FuncFormatter(scientific)

class CSHNetGraph:
    '''
    directed:True 有向图，False 无向图
    '''
    def __init__(self,directed = True):
        self.m_directed = directed
        self.m_nodes = []
        self.m_edges = []
        self.m_node_id = []
        
    def Add_Node(self,label,value,title,group=None,size=None,color='#00ff1e',node_id = None):
        tmp = {}
        if not node_id:
            node_id = len(self.m_nodes) + 1
        tmp['id'] = node_id
        tmp['label'] = label
        tmp['value'] = value
        tmp['title'] = title
        tmp['group'] = group
        tmp['size'] = size
        tmp['color'] = color
        if not node_id in self.m_node_id:
            self.m_nodes.append(tmp)
            self.m_node_id.append(node_id)
        return node_id
    
    def Add_Edge(self,from_id,to_id,weight,label):
        tmp = {}
        tmp['from'] = from_id
        tmp['to'] = to_id
        tmp['weight'] = weight
        tmp['label'] = label
        self.m_edges.append(tmp)
                
    def Show(self,file_name = 'pyvis.html'):
        net = Network('1024px','1024px',directed=self.m_directed)
        for node in self.m_nodes:
            net.add_node(node['id'],
                         label=node['label'],
                         value=node['value'],
                         color=node['color'],
                         title=node['title'],
                         group=node['group'])
        for edge in self.m_edges:
            net.add_edge(edge['from'], 
                         edge['to'], 
                         weight=edge['weight'],
                         label=edge['label'])
        net.toggle_physics(False)
        #net.show_buttons(filter_=['physics'])
        print(file_name)
        net.show(file_name)
    
    def Add(self,from_node,to_node,title=None,weight=None):
        node_id = None if 'id' not in from_node else from_node['id']
        label = None if 'label' not in from_node else from_node['label']
        value = None if 'value' not in from_node else from_node['value']
        title = None if 'title' not in from_node else from_node['title']
        group = None if 'group' not in from_node else from_node['group']
        size = None if 'size' not in from_node else from_node['size']
        color = '#00ff1e' if 'color' not in from_node else from_node['color']
        id_from = self.Add_Node(label=label,value=value,title=title,group=group,size=size,color=color,node_id=node_id)
       
        node_id = None if 'id' not in to_node else to_node['id']
        label = None if 'label' not in to_node else to_node['label']
        value = None if 'value' not in to_node else to_node['value']
        title = None if 'title' not in to_node else to_node['title']
        group = None if 'group' not in to_node else to_node['group']
        size = None if 'size' not in to_node else to_node['size']
        color = '#00ff1e' if 'color' not in to_node else to_node['color']
        id_to = self.Add_Node(label=label,value=value,title=title,group=group,size=size,color=color,node_id=node_id)
        
        self.Add_Edge(id_from,id_to,weight,title)
        return id_from,id_to

    @staticmethod
    def call_edge_info(from_node,to_node):
        weight = max(from_node['value'],to_node['value'])
        title = '%d:%d'%(from_node['value'],to_node['value'])
        return weight,title
    '''
    可视化时序数据
    df_in:时序数据，必须的字段有label(节点的标签),value(节点的值),title(节点的tips)
    call_edge_info: callback根据from和to节点，计算edge的title和weight
    file_name：输出的html文件
    '''
    @staticmethod
    def Show_Series(df_in,call_edge_info = None,file_name = 'pyvis.html'):
        net = CSHNetGraph()
        for item in json.loads(df_in.to_json(orient='records')):
            label = None if 'label' not in item else item['label']
            value = None if 'value' not in item else item['value']
            title = None if 'title' not in item else item['title']
            group = None if 'group' not in item else item['group']
            size = None if 'size' not in item else item['size']
            color = '#00ff1e' if 'color' not in item else item['color']
            net.Add_Node(label=label,value=value,title=title,group=group,size=size,color=color)
   
        for i in range(df_in.shape[0]-1):
            curr_node = json.loads(df_in.iloc[i].to_json())
            next_node = json.loads(df_in.iloc[i+1].to_json())
            if call_edge_info:
                weight,title = call_edge_info(curr_node,next_node)
            else:
                weight,title = CSHNetGraph.call_edge_info(curr_node,next_node)
            net.Add_Edge(i+1,i+2,weight,title)
            
        net.Show(file_name)
        
class CSHVisualize:
    
    def __init__(self):
        pass

    '''
    绘制热图
    '''
    @staticmethod
    def show_heatmap(df,title="heatmap demo"):
        #plt.ticklabel_format(style='plain', axis='both')
        sns.set(font_scale=1.5)
        cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
        sns.heatmap(df, cmap=cmap, linewidths=0.05, annot=False, fmt="g", annot_kws={"fontsize": 32})
        plt.xticks(rotation=60)
        plt.yticks(rotation=15)
    '''
    绘制桑基图
    1.节点，边，流量
    2.nodes中的"name"不要重命, 否则会不识别, 导致没有图
    3.links中source和target对应的值一定要在nodes的name对应的值中, 否则图中会不显示(但是不会报错)
    4.links中source和target对应的值不能相同, 否则图不显示
    '''
    @staticmethod
    def get_sankey(df_data_list,title="sankey demo"):
        if type(df_data_list) == list:
            return CSHVisualize.get_sankey_base(df_data_list,title)
        else:
            df_tmp =  df_data_list
            columns = df_tmp.columns.tolist()
            df_list = []    
            value_key = columns[-1]
            for i in range(1,len(columns) - 1):
                groupList = []
                name1 = columns[i-1]
                name2 = columns[i]
                for j in range(0,i+1):
                    name = columns[j]
                    groupList.append(name)
                df_t = df_tmp.groupby(groupList).sum()[value_key].reset_index()
                df_values = pd.DataFrame()
                df_values[name1] = df_t[name1]
                df_values[name2] = df_t[name2]
                df_values[value_key] = df_t[value_key]
                df_list.append(df_values)
                
            return CSHVisualize.get_sankey_base(df_list,title)
        
    @staticmethod
    def get_sankey_base(df_data_list,title="sankey demo"):
        def get_sankey_data(df_in):
            def get_name(feild,value):
                #return str(value)
                return str(feild)+":"+str(value)
            #create nodes
            columns = df_in.columns.tolist()
            nodes = []
            for item in df_in[columns[0]].unique():
                dic = {}
                dic['name'] = get_name(columns[0],item)
                nodes.append(dic)
            for item in df_in[columns[1]].unique():
                dic = {}
                dic['name'] = get_name(columns[1],item)
                nodes.append(dic)
            #create links
            links = []
            for item in df_in.values:
                dic = {}
                dic['source'] = get_name(columns[0],item[0])
                dic['target'] = get_name(columns[1],item[1])
                dic['value'] = float(item[2])
                links.append(dic)
            df_tmp = pd.DataFrame(nodes)
            #df_tmp = df_tmp.sort_values(by='value',ascending=False)
            nodes = json.loads(df_tmp.to_json(orient='records'))
            return nodes,links
        nodes,links = [],[]
        if type(df_data_list) != list:
            nodes,links = get_sankey_data(df_data_list)
        else:         
            for df in df_data_list:
                nodes1,links1 = get_sankey_data(df)
                for n in nodes1:
                    if not n in nodes:
                        nodes.append(n)
                for l in links1:
                    links.append(l)
                
        graph = Sankey(init_opts=opts.InitOpts(width="1024", height="768px",theme='westeros'))
        graph.add(title,nodes=nodes,links=links,
              linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
              label_opts=opts.LabelOpts(position="right"))
        return graph
    '''
    显示等高线
    df_data：待显示数据，格式为（x，y，value）
    isBlack: 黑白显示
    withTips：是否显示数字
    '''
    @staticmethod
    def show_contour(df_data, isBlack=True, withTips = True):
        columns = df_data.columns.tolist()
        df_tmp = df_data.copy(deep = True)
        category_map = {}
        for i in range(2):
            key = columns[i]
            key_type = str(df_tmp[key].dtypes)
            if key_type == "object":
                df_tmp[key] = df_tmp[key].astype("category")
                category_map[key] = dict(zip(df_tmp[key].cat.codes,df_tmp[key]))            
            key_type = str(df_tmp[key].dtypes)
            if key_type == "category":
                df_tmp[key] = df_tmp[key].cat.codes
                
        hdfpivot = df_tmp.pivot(columns[0], columns[1], columns[2])
        X = hdfpivot.columns.values
        Y = hdfpivot.index.values
        Z = hdfpivot.values
        x, y = np.meshgrid(X, Y)
        #plt.ticklabel_format(style='plain', axis='both')
        cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
        if isBlack:
            contours = plt.contour(x, y, Z, 3, colors='black')  # the NAN will be plotted as white spaces
        else:
            contours = plt.contourf(x, y, Z ,cmap=cmap, linewidths=0.05 ) #the NAN will be plotted as white spaces
        if withTips:
            plt.clabel(contours, inline=True, fontsize=8)
        plt.colorbar()
        return category_map
    '''
    可视化时序数据
    df_in:时序数据，必须的字段有label(节点的标签),value(节点的值),title(节点的tips)
    call_edge_info: callback根据from和to节点，计算edge的title和weight
    file_name：输出的html文件
    '''
    @staticmethod
    def show_sequence(df_data, call_edge_info = None,file_name = 'pyvis.html'):
        CSHNetGraph().Show_Series(df_data,call_edge_info,file_name)

def test_heatmap():
    data_list = []
    data_list.append({"类别1": "A1", "类别2": "B1", "Count": 100})
    data_list.append({"类别1": "A2", "类别2": "B2", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B1", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B2", "Count": 10})
    df_data = pd.DataFrame(data_list)
    pt = df_data.pivot_table(values='Count', index=["类别1"], columns=['类别2'], aggfunc=np.sum, fill_value=0)
    CSHVisualize.show_heatmap(pt)

def test_sankey():
    data_list = []
    data_list.append({"类别1": "A1", "类别2": "B1", "Count": 100})
    data_list.append({"类别1": "A2", "类别2": "B2", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B1", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B2", "Count": 10})
    df_data = pd.DataFrame(data_list)
    CSHVisualize.get_sankey(df_data).render_notebook()

def test_contour():
    data_list = []
    for i in range(10):
        for j in range(10):
            tmp = {"x": i, "y": j, "Count": np.sin(i * j)}
            data_list.append(tmp)
    df_data = pd.DataFrame(data_list)
    CSHVisualize.show_contour(df_data)
    
def test_sequence():
    data = []
    for i in range(5):
        tmp = {}
        tmp['label'] = "name-%d"%i
        tmp['value'] = i
        tmp['title'] = "title-%d"%i

        data.append(tmp)
    df_data = pd.DataFrame(data)
    CSHVisualize.show_sequence(df_data)
    
def main():
    test_heatmap()
    test_sankey()
    test_contour()
    test_sequence()
   
if __name__ == "__main__":
    main()
