from datasets import DatasetDict, Dataset,load_dataset
import json,gc,os,sys,random,warnings,torch
import pandas as pd
import numpy as np
import logging as logger
from transformers import DataCollatorWithPadding,AutoTokenizer,DataCollatorForSeq2Seq,PreTrainedTokenizer
sys.path.append("../")
from Config import g_default_max_length
warnings.filterwarnings("ignore")

class CLLMDataset(object):
    
    def __init__(self):
        self.m_data = None
        
    def load_from_huggingface(self,dataset_name):
        self.m_data = load_dataset(dataset_name)
        def rename_column(example):
            example['labels'] = example.pop('label')
            return example
        
        #将标签字段label，转换为labels
        for split in self.m_data:
            if 'label' in self.m_data[split].column_names:
                self.m_data = self.m_data.map(rename_column)
                break

    def load_from_pandas(self,df_train,df_verify,df_test=pd.DataFrame()):
        dataset_train = Dataset.from_pandas(df_train.reset_index(drop=True))
        dataset_test = Dataset.from_pandas(df_test.reset_index(drop=True))
        dataset_val = Dataset.from_pandas(df_verify.reset_index(drop=True))
        if df_test.shape[0] > 0:
            self.m_data = DatasetDict({'train': dataset_train,'val': dataset_val,'test': dataset_test})
        else:
            self.m_data = DatasetDict({'train': dataset_train,'val': dataset_val})

    def get_dataset(self):
        return self.m_data
    
    def get_train(self):
        return pd.DataFrame(self.m_data['train'])
    
    def get_test(self):
        return pd.DataFrame(self.m_data['test'])
    
    def get_verify(self):        
        if "val" in self.m_data:
            return pd.DataFrame(self.m_data['val'])
        else:
            return pd.DataFrame()
    
    def check_data_valid(self,tokenizer,max_length=g_default_max_length):
        all_data = []
        for split, dataset in self.m_data.items():
            print(f"Processing {split} split:")
            for record in dataset:
                tmp = record
                try:
                    tokenizer(record['text'],truncation=True, max_length=max_length)
                    tmp['valid'] = True
                except:
                    tmp['valid'] = False
                    logger.exception("check_data_valid exception")

                tmp['set'] = split
                all_data.append(tmp)
        return pd.DataFrame(all_data)

    def get_single_classify_tokenized_data(self,tokenizer,max_length=g_default_max_length):
        def data_preprocesing(row):
            return tokenizer(row['text'], padding=True, truncation=True, max_length=max_length)

        tokenized_data = self.m_data.map(data_preprocesing,batched=True, remove_columns=['text'])
        tokenized_data.set_format("torch")
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
        return tokenized_data,collate_fn
    
    def get_mul_classify_tokenized_data(self,tokenizer, max_length=g_default_max_length):
        def preprocess_function(example):
            inputs,labels = example['text'],example['labels']
            input_ids,attention_mask,label_ids = [],[],[]
            for input_text, label in zip(inputs, labels):
                if isinstance(label, list):
                    target_labels = label
                else:
                    target_labels = json.loads(label)  
                encoded_input = tokenizer(input_text, max_length=max_length, truncation=True, padding="max_length", add_special_tokens=True)
                input_ids.append(encoded_input["input_ids"])
                attention_mask.append(encoded_input["attention_mask"])
                label_ids.append(target_labels)
            return {"input_ids": input_ids,"attention_mask": attention_mask,"labels": label_ids}

        # 对数据集进行 map 操作
        tokenized_data = self.m_data.map(preprocess_function, batched=True)
        tokenized_data.set_format("torch")
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
        return tokenized_data, collate_fn

    def get_generate_tokenized_data(self,tokenizer,max_length=g_default_max_length):
        return self.get_qa_tokenized_data(tokenizer,max_length)

    def get_qa_tokenized_data(self,tokenizer,max_length=g_default_max_length):
        def preprocess_function(example):
            inputs,labels = example['text'],example['labels']
            targets = labels
            input_ids,attention_mask,label_ids = [],[],[]
            for input_text, target in zip(inputs, targets):
                instruction = tokenizer(input_text, max_length=max_length, truncation=True, padding="max_length", add_special_tokens=False)
                response = tokenizer(target, max_length=max_length, truncation=True, padding="max_length", add_special_tokens=False)
                # 合并 input_ids 和 attention_mask
                combined_input_ids = instruction["input_ids"] + response["input_ids"]
                combined_attention_mask = instruction["attention_mask"] + response["attention_mask"]
                # 对应的标签处理,-100表示不计算损失
                combined_labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
                # 添加 padding
                combined_input_ids += [tokenizer.pad_token_id] * (max_length - len(combined_input_ids))
                combined_attention_mask += [0] * (max_length - len(combined_attention_mask))
                combined_labels += [-100] * (max_length - len(combined_labels))
                # 截断以符合 max_length
                input_ids.append(combined_input_ids[:max_length])
                attention_mask.append(combined_attention_mask[:max_length])
                label_ids.append(combined_labels[:max_length])
            return {"input_ids": input_ids,"attention_mask": attention_mask,"labels": label_ids}

        # 对数据集进行 map 操作
        tokenized_data = self.m_data.map(preprocess_function, batched=True)
        tokenized_data.set_format("torch")
        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
        return tokenized_data, collate_fn

    def get_class_weights(self):
        ds_label = pd.Series(self.m_data['train']['labels'])
        class_weights=(1/ds_label.value_counts(normalize=True).sort_index()).tolist()
        class_weights=torch.tensor(class_weights)
        class_weights=class_weights/class_weights.sum()
        return class_weights

def main():
    data = CLLMDataset()
    data.load_from_huggingface("ag_news")
    df_train,df_test = data.get_train(), data.get_test()
    weights = data.get_class_weights()
    print(weights)

if __name__ == "__main__":
    main()
