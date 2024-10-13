import gc,os,random,warnings,sys,json,torch
sys.path.append("./ResNet/")
sys.path.append("./common/")
sys.path.append("./share/")
from torch.utils.data import Dataset, DataLoader,random_split,Subset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from Config import g_data_root,g_embedding_shape
g_embedding_root = "%s/embedding/"%(g_data_root)

class CEmbeddingDataset(Dataset):
    
    def __init__(self,index_file = None):
        if index_file:
            self.df_index = self.load(index_file)

    def load(self,index_file):
        self.df_index = pd.read_csv(index_file,index_col=0).reset_index(drop=True)
        return self.df_index
        
    def preprocess(self,data):
        return data
        #data_transer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #return data_transer(data)

    def __len__(self):
        return self.df_index.shape[0]

    def __getitem__(self, idx):
        file_name = self.df_index.iloc[idx]['file']
        labels = self.df_index.iloc[idx]['labels']
        with open(file_name,"r") as fp:
            data = np.array(json.loads(fp.read())).reshape(-1,g_embedding_shape[0],g_embedding_shape[1])
        data = torch.from_numpy(data).to(torch.float32)
        data = self.preprocess(data)
        return data, torch.tensor(labels)
        
    @staticmethod
    def get_loader(full_dataset,test_ratio = 0.2,batch_size=16):
        train_ratio = 1- test_ratio
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        train_dataset = Subset(full_dataset, range(train_size))
        test_dataset = Subset(full_dataset, range(train_size, len(full_dataset)))
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader,test_loader

class CDLResNet():
    
    def __init__(self):
        self.m_model = None
        self.m_optimizer = None
        self.m_scheduler = None
        self.m_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build(self,n_class):
        #self.m_model = models.resnet18(pretrained=False)
        self.m_model = models.resnet50(pretrained=False)
        #self.m_model = models.resnet101(pretrained=False)
        # 修改第一层卷积层，将输入通道数从 3 改为 1
        self.m_model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后一层，为输出的类别数
        self.m_model.fc = nn.Linear(self.m_model.fc.in_features, n_class)
        self.m_criterion = nn.CrossEntropyLoss()
        self.m_optimizer = optim.Adam(self.m_model.parameters(), lr=0.001)
        #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #self.m_optimizer = optim.Adam(self.m_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
        self.m_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.m_optimizer, factor = 0.1, patience=5)
        self.m_model = self.m_model.to(self.m_device)

    def save(self,file_name):
        torch.save(self.m_model, file_name)

    def load(self,file_name):
        self.m_model = torch.load(file_name)
        self.m_model.eval()  
        torch.no_grad()
        
    def train(self,train_loader,n_epoch = 20):
        for epoch in tqdm(range(n_epoch),desc="Total"):
            self.m_model.train()
            for embedding, labels in train_loader:
                embedding, labels = embedding.to(self.m_device), labels.to(self.m_device)
                self.m_optimizer.zero_grad()
                outputs = self.m_model(embedding)
                loss = self.m_criterion(outputs, labels)
                loss.backward()
                self.m_optimizer.step()        
            msg = (f"Epoch [{epoch+1}/{n_epoch}], Loss: {loss.item():.6f}")
            print(msg)
            
    def predict(self,embeddling):
        data = embeddling.to(self.m_device)
        outputs = self.m_model(data)
        _, predicted = torch.max(outputs, 1)
        return predicted

def main():
    attack = "Backdoor_attack"
    embedding_index = "%s/text-label/%s/index.csv"%(g_embedding_root,attack)
    dataset = CEmbeddingDataset(embedding_index)
    train_loader,test_loader = CEmbeddingDataset.get_loader(dataset,test_ratio=0.2,batch_size=32)
    for batch_idx, (data_batch, labels_batch) in enumerate(train_loader):
        print('Labels:', labels_batch)
        break

    model = CDLResNet()
    model.build(2)
    model.train(train_loader)
    model.save("test.pth")
    
    model1 = CDLResNet()
    model1.load("test.pth")
    for batch_idx, (data_batch, labels_batch) in enumerate(test_loader):
        predicted = model1.predict(data_batch)
        print(batch_idx,labels_batch)
        print(batch_idx,predicted)
        
if __name__ == "__main__":
    main()
