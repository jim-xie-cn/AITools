import io,os,json,configparser,gc,warnings
import pandas as pd
import torch
'''
Pytorch系统操作
'''
class CPTCommon():
    '''
    设置并行线程数
    n_thread : 线程数
    '''
    @staticmethod
    def set_mul_thread(n_thread = 10 ):
        
        if torch.get_num_interop_threads() != n_thread:
            torch.set_num_interop_threads(n_thread) # Inter-op parallelism
        
        if torch.get_num_threads() != n_thread:
            torch.set_num_threads(n_thread) # Intra-op parallelism
    
    '''
    读取ini配置参数
    file_name : 配置文件名
    section : ini的段
    key : ini的key
    '''    
    @staticmethod 
    def read_config(file_name,section,key ):
        config = configparser.ConfigParser()
        config.read(file_name)
        return config[section][key]

def main():
    CPTCommon.set_mul_thread( n_thread = 5 )
    model_name_path = CPTCommon.read_config("../configuration/config.ini",
                                            "transformers",
                                            "model_name_or_path")
    print(model_name_path.strip())

if __name__ == '__main__':
    main()
