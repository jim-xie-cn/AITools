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
from PTDataset import CPTDataset
warnings.filterwarnings('always')
#pytorch内存泄露
#https://medium.com/@raghadalghonaim/memory-leakage-with-pytorch-23f15203faa4

'''
训练处理
'''
class CPTTraining:
    '''
    初始化训练对象
    data_loader : 分批次的数据集
    model: 模型
    optimizer:优化器
    scheduler:定时处理对象
    device:GPU或CPU
    '''
    def __init__(self,
                 data_loader,
                 model,
                 optimizer,
                 scheduler,
                 device = 'cpu'):
        
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    
    '''
    进行一轮训练（多个批次训练）
    clip_grad：是否归一化grad
    返回值
    avg_epoch_loss：多个批次的平均损失
    predictions_labels：预测标签
    ground_true_labels：真实标签
    '''
    def train(self, clip_grad = True):
        
        self.model.train()
        losses = 0
        predictions_labels = []
        ground_true_labels = []

        for batch in tqdm(self.data_loader,total = len(self.data_loader),position=0, leave=True):
            
            ground_true_labels += batch['labels'].numpy().flatten().tolist()
            
            batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
            
            self.model.zero_grad()

            outputs = self.model(**batch)
            loss,logits = outputs[:2]

            losses+= loss.detach().item()
            loss.backward()
            
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

            self.optimizer.step()

            self.scheduler.step()

            logits = logits.detach().cpu().numpy()

            predictions_labels += logits.argmax(axis = -1).flatten().tolist()
                        
        avg_epoch_loss = losses / len(self.data_loader)

        return avg_epoch_loss,predictions_labels,ground_true_labels

'''
Verification操作
'''
class CPTValidation:
    '''
    初始化训练对象
    data_loader : 分批次的数据集
    model: 模型
    device:GPU或CPU
    '''
    def __init__(self, data_loader, model, device = 'cpu' ):
        
        self.data_loader = data_loader
        self.model = model
        self.device = device
    '''
    进行一轮校验（多个批次校验）
    返回值
    avg_epoch_loss：多个批次的平均损失
    predictions_labels：预测标签
    ground_true_labels：真实标签
    '''
    def validate(self):
        
        self.model.eval()
        losses = 0
        correct_predictions = 0
        predictions_labels = []
        ground_true_labels = []

        for batch in tqdm(self.data_loader,total = len(self.data_loader),position=0, leave=True):
            
            ground_true_labels += batch['labels'].numpy().flatten().tolist()
            batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}

            with torch.no_grad():

                outputs = self.model(**batch)
                loss,logits = outputs[:2]

                losses+= loss.detach().item()
                
                logits = logits.detach().cpu().numpy()

                predictions_labels += logits.argmax(axis = -1).flatten().tolist()

        avg_epoch_loss = losses / len(self.data_loader)

        return avg_epoch_loss,predictions_labels,ground_true_labels  

'''
Fine tune操作类
'''
class CPTFineTuning:
    
    def __init__(self,
                 tokenizer,
                 model,
                 n_epochs = 3,
                 device='cpu',
                 n_thread = 8):
        
        CPTCommon.set_mul_thread(n_thread)
        self.m_tokenizer = tokenizer
        self.m_model = model
        self.m_model.to(device)

        self.m_device = device
        self.m_train_loader = None
        self.m_verify_loader = None
        self.m_optimizer = None
        self.m_scheduler = None
        self.m_epochs = n_epochs
        
        self.m_train_samples = 0
        self.m_callback = None
        self.m_model_path = None
    
    def set_dataset(self,train_set,verify_set, n_batch ):
        self.m_train_samples = len(train_set)
        self.m_train_loader = train_set.get_loader(n_batch = n_batch )
        self.m_verify_loader = verify_set.get_loader(n_batch = n_batch )

    def set_optimizer(self,optimizer = None):
        if optimizer is None:
            self.m_optimizer = torch.optim.AdamW(self.m_model.parameters(),lr = 2e-5,eps = 1e-8)
        else:
            self.m_optimizer = optimizer
        
    def set_scheduler(self,scheduler=None):
        
        total_steps = self.m_train_samples * self.m_epochs

        if scheduler is None:
            self.m_scheduler = get_linear_schedule_with_warmup(self.m_optimizer,
                                                               num_warmup_steps = 0,
                                                               num_training_steps= total_steps)
        else:
            self.m_scheduler = scheduler
            
    @staticmethod
    def monitor_callback(epoch,result):
        training_acc = accuracy_score(result['train_gt'],result['train_predict'])
        validation_acc = accuracy_score(result['validation_gt'],result['validation_predict'])

        print("Epoch %d: train loss %f train acc %f, val loss %f val acc %f"%
              (epoch,result['train_loss'],training_acc,result['validation_loss'],validation_acc) )

    def set_monitor(self, callback = None, model_path = None):
        if callback is None:
            self.m_callback  = CPTFineTuning.monitor_callback
        else:
            self.m_callback = callback
            
        self.m_model_path = model_path
        
    def train(self):
        trainer = CPTTraining(data_loader = self.m_train_loader,
                              model = self.m_model,
                              optimizer = self.m_optimizer,
                              scheduler = self.m_scheduler,
                              device = self.m_device)
        
        validator = CPTValidation(data_loader = self.m_verify_loader,
                                  model = self.m_model,
                                  device = self.m_device)
        
        for epoch in tqdm(range(self.m_epochs),position=0, leave=True):
            result = {}
            training_loss,prediction_lables,ground_true_labels = trainer.train()
            result['train_loss'] = training_loss
            result['train_predict'] = prediction_lables
            result['train_gt'] = ground_true_labels

            validation_loss,prediction_lables,ground_true_labels = validator.validate()            
            result['validation_loss'] = validation_loss
            result['validation_predict'] = prediction_lables
            result['validation_gt'] = ground_true_labels
            
            if self.m_model_path is not None:
                model_file = "%s/checkpoint-%d.bin"%(self.m_model_path,epoch)
                torch.save(self.m_model.state_dict(),model_file)
                
            if self.m_callback is not None:
                self.m_callback(epoch,result)
            
            gc.collect()

def main():
    
    model_name_path = CPTCommon.read_config("../configuration/config.ini",'transformers','model_name_or_path').strip()

    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_path, num_labels = 2)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_name_path)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_name_path, config = model_config)

    dataset = CPTDataset(tokenizer)

    dataset.read_csv('../dataset/movies.csv',"review","sentiment",test_samples=10)

    ft = CPTFineTuning(tokenizer, model, n_epochs=1 )

    train_set = dataset
    validate_set = dataset
    ft.set_dataset(train_set, validate_set, batch_size = 4)

    ft.set_optimizer()

    ft.set_scheduler()

    ft.set_monitor()

    ft.train()

if __name__ == '__main__':
    main()
