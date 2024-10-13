from datasets import DatasetDict, Dataset,load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model,PeftModel, PeftConfig
from transformers import Trainer,DataCollatorWithPadding,AutoModelForSequenceClassification
import torch.nn.functional as F
import gc,torch,os,warnings,random
import pandas as pd
from tqdm import tqdm
import numpy as np
from LLMCommon import CLLMCommon
from LLMDataset import CLLMDataset
from LLMBase import CLLMBase
from Config import quantization_config,lora_config_classify,training_config,g_model_root,g_model_base,g_data_root,g_default_max_length,g_default_batch_size
np.set_printoptions(suppress=True, precision=6)
warnings.filterwarnings("ignore")

class CustomRegressionTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        predictions = outputs.get('logits').squeeze()
        #mean_x = torch.mean(predictions)
        #variance = F.mse_loss(predictions, mean_x.expand_as(predictions))
        #loss = F.mse_loss(predictions, labels) +  torch.sqrt(variance)
        #loss = F.l1_loss(predictions, labels)
        #loss = F.huber_loss(predictions, labels, delta=1.0) 
        loss = F.mse_loss(predictions, labels)
        return (loss, outputs) if return_outputs else loss 
 
class CLLMRegression(object):
    
    def __init__(self):
        self.m_num_labels = 1
        self.m_model = CLLMBase(task='regression')

    def get_custom_tokens(self):
        feature_token_file = "%s/features/features_token.csv"%g_data_root
        if os.path.exists(feature_token_file):
            df_feature_token = pd.read_csv(feature_token_file)
            custom_tokens = df_feature_token['feature'].unique().tolist()
            return custom_tokens
        return []

    def save(self,model_directory = None):
        if model_directory == None:
            model_directory = "%s"%g_model_root
        print("mode save to ",model_directory)
        self.m_model.save(model_directory)

    def load_raw(self,quantization_config = None):
        self.m_model.load(model_type="raw", model_name_or_path = g_model_base, quantization_config = quantization_config , num_labels = self.m_num_labels)
        custom_tokens = self.get_custom_tokens()
        if custom_tokens:
            self.m_model.add_tokens(custom_tokens)

    def load_ft(self,quantization_config = None):
        model_directory = "%s"%g_model_root
        self.m_model.load(model_type = "ft", model_name_or_path = model_directory, quantization_config = quantization_config, num_labels = self.m_num_labels)
        custom_tokens = self.get_custom_tokens()
        if custom_tokens:
            self.m_model.add_tokens(custom_tokens)

    def train(self,llm_dataset,max_length=g_default_max_length):
        tokenizer = self.m_model.get_tokenizer()
        model = self.m_model.get_model()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config_classify)

        tokenized_data,collate_fn=llm_dataset.get_single_classify_tokenized_data(tokenizer,max_length=max_length)

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            labels = labels.reshape(-1, 1)
            return CLLMCommon.evalulate_regression(labels, preds)

        trainer = CustomRegressionTrainer(model = model,
                                          args = training_config,
                                          train_dataset = tokenized_data['train'],
                                          eval_dataset = tokenized_data['val'],
                                          tokenizer = tokenizer,
                                          data_collator = collate_fn,
                                          compute_metrics = compute_metrics)#class_weights=self.m_llm_dataset.get_class_weights()
        
        self.m_model.set_trainer(trainer)

        return self.m_model.get_trainer().train()

    def predict(self,sentences, batch_size = g_default_batch_size, max_length=g_default_max_length):
        max_records_per_df = batch_size
        num_splits = int(np.ceil(len(sentences) / max_records_per_df))
        sub_dfs = np.array_split(sentences, num_splits)
        result,probability = [],[]
        for sub_sentence in tqdm(sub_dfs,desc="predicting"):
            sub_sentence = sub_sentence.tolist()
            inputs = self.m_model.get_input(sub_sentence,max_length=max_length)
            outputs = self.m_model.predict(inputs,max_length = max_length)
            pred = outputs.squeeze().cpu().numpy().tolist()
            result.extend(pred)
            probability.extend(outputs.cpu().numpy().tolist())

        return result,probability

def main():
    CLLMCommon.init_random()
    #prepare data
    dataset = CLLMDataset()
    dataset.load_from_huggingface("ag_news")
    df_train,df_verify,df_test = dataset.get_train().sample(200),dataset.get_train().tail(100), dataset.get_test().head(50)
    df_train['labels'] = df_train['labels'].astype(float)
    df_test['labels'] = df_test['labels'].astype(float)
    df_verify['labels'] = df_verify['labels'].astype(float)
    df_train['labels'] = (df_train['labels'] - df_train['labels'].mean())/df_train['labels'].std()
    df_verify['labels'] = (df_verify['labels'] - df_verify['labels'].mean())/df_verify['labels'].std()
    df_test['labels'] = (df_test['labels'] - df_test['labels'].mean())/df_test['labels'].std()
    dataset.load_from_pandas(df_train,df_verify,df_test)

    sentences = df_train.text.tolist()
    labels = df_train.labels.tolist()
    
    print("Begin train model")
    ft = CLLMRegression()
    ft.load_raw( quantization_config = quantization_config )
    result = ft.train(dataset,max_length=512)
    print(result)
    ft.save()
    del ft
    
    print("Begin test a fine tuned model")
    ft1 = CLLMRegression()
    ft1.load_ft( quantization_config = quantization_config )
    predicts,probability = ft1.predict(sentences)
    print(labels,predicts)

if __name__ == "__main__":
    main()
