import io,os,json,configparser,gc,warnings
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset,DataLoader
from ml_things import fix_text # from ftfy import fix_text
from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          set_seed)
from sklearn.metrics import accuracy_score
from PTCommon import CPTCommon
from SHCommon import CSHCommon
warnings.filterwarnings('always')

'''
Pytorch数据集
'''
class CPTDataset(Dataset):
    '''
    初始化计数据集对象
    tokenizer : 分词对象
    max_length: 最大长度
    '''
    def __init__(self,tokenizer,max_length=None):
        self.m_tokenizer = tokenizer
        self.m_max_seq_len = tokenizer.model_max_length if max_length is None else max_length

        self.inputs = None
        self.sequence_len = 0
        self.n_examples = 0
    '''
    数据集内置函数，返回数据集大小
    '''
    def __len__(self):
        return self.n_examples
    '''
    数据集内置函数，返回数据集内容
    '''
    def __getitem__(self, item):
        return {key: self.inputs[key][item] for key in self.inputs.keys()}
    '''
    将数据集转化为多批次的，DataLoader对象。
    batch_size: 每个批次的样本数
    shuffle: 是否要shuffle
    drop_last: 当最后样本数量不够一个批次时，是否要丢弃
    返回：DataLoader对象
    '''
    def get_loader(self,n_batch,shuffle=False,drop_last=False):
        return DataLoader(self,batch_size=n_batch,shuffle=shuffle,drop_last= drop_last)
    '''
    根据样本和标签，将文本进行分词后，生成数据集
    texts: 文本列表
    labels: 标签列表
    '''
    def load(self,
             texts,
             labels,
             add_special_tokens=True,
             truncation=True,
             padding=True, 
             return_tensors='pt'):
        
        self.n_examples = len(labels)

        print('Tokenizing all texts....')
        self.inputs = self.m_tokenizer(texts,
                                       add_special_tokens=add_special_tokens, 
                                       truncation=truncation, 
                                       padding=padding, 
                                       return_tensors=return_tensors, 
                                       max_length = self.m_max_seq_len)

        self.sequence_len = self.inputs['input_ids'].shape[-1]
        print(f'Max Sequence Length of Texts after Tokenization: {self.sequence_len}')
        
        self.inputs.update({'labels': torch.tensor(labels)})
        
        print('finished')
    '''
    从csv文件中，读取文本和标签，并将文本进行分词，生成数据集
    file_name: csv文件
    col_name_text:文本的列名
    col_name_label:标签的列名
    label_map:文本标签，和数组标签的对应关系
    test_samples:取前若干个样本，为0表示取所有样本
    '''
    def read_csv(self,
                 file_name,
                 col_name_text,
                 col_name_label,
                 label_map = None,
                 add_special_tokens=True,
                 truncation=True,
                 padding=True, 
                 return_tensors='pt',
                 test_samples = 0):
        
        df =  CSHCommon.load_csv(file_name)
        if test_samples > 0:
            df = df.head( test_samples )
        
        def format_data(x):
            ret = x
            ret[col_name_text] = fix_text(ret[col_name_text])
            if label_map:
                ret[col_name_label] = label_map[ret[col_name_label]]
            return ret
        
        tqdm.pandas(desc='format string and lables')
        df = df.progress_apply(lambda x:format_data(x),axis=1)
        
        texts = df[col_name_text].tolist()
        labels = df[col_name_label].tolist()
            
        self.load(texts,
                  labels,
                  add_special_tokens=add_special_tokens, 
                  truncation=truncation, 
                  padding=padding, 
                  return_tensors=return_tensors)

def main():
    model_name_path = CPTCommon.read_config("../configuration/config.ini",'transformers','model_name_or_path').strip()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_name_path)
    dataset = CPTDataset(tokenizer)
    dataset.read_csv('../dataset/movies.csv',"review","sentiment",test_samples=10,label_map={'negative': 0, 'positive': 1})
    print(len(dataset))

if __name__ == '__main__':
    main()
