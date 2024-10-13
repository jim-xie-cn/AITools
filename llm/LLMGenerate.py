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
from Config import quantization_config,lora_config_generate,training_config,g_model_root,g_model_base
from Config import g_data_root,g_default_max_length,g_default_batch_size,g_default_max_new_tokens
from trl import SFTTrainer
np.set_printoptions(suppress=True, precision=6)
warnings.filterwarnings("ignore")

class CLLMLLMGenerate(object):
    
    def __init__(self):
        self.m_num_labels = 0
        self.m_model = CLLMBase(task='generate')

    def save(self,model_directory = None):
        if model_directory == None:
            model_directory = "%s"%g_model_root
        print("mode save to ",model_directory)
        self.m_model.save(model_directory)

    def get_custom_tokens(self):
        feature_token_file = "%s/features/features_token.csv"%g_data_root
        if os.path.exists(feature_token_file):
            df_feature_token = pd.read_csv(feature_token_file)
            custom_tokens = df_feature_token['feature'].unique().tolist()
            return custom_tokens
        return []

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
        model = get_peft_model(model, lora_config_generate)
        tokenized_data,collate_fn=llm_dataset.get_generate_tokenized_data(tokenizer,max_length=max_length)

        trainer = SFTTrainer(model=model,
                             train_dataset=tokenized_data['train'],
                             eval_dataset = tokenized_data['val'],
                             tokenizer = tokenizer,
                             peft_config=lora_config_generate,
                             dataset_text_field="text",
                             max_seq_length=None,
                             args=training_config,
                             packing=False)

        self.m_model.set_trainer(trainer)

        return self.m_model.get_trainer().train()

    def predict(self,sentences, batch_size = g_default_batch_size, max_new_tokens = g_default_max_new_tokens, max_length=g_default_max_length):
        max_records_per_df = batch_size
        num_splits = int(np.ceil(len(sentences) / max_records_per_df))
        sub_dfs = np.array_split(sentences, num_splits)
        result = []
        for sub_sentence in tqdm(sub_dfs,desc="predicting"):
            sub_sentence = sub_sentence.tolist()
            inputs = self.m_model.get_input(sub_sentence, max_length = max_length)
            outputs = self.m_model.generate(inputs, max_new_tokens = max_new_tokens, max_length = max_length)
            pred = outputs
            result.extend(pred)
        return result

def main():
    CLLMCommon.init_random()
    #prepare data
    dataset = CLLMDataset()
    dataset.load_from_huggingface("mlabonne/guanaco-llama2-1k")
    df_train,df_verify,df_test = dataset.get_train().head(50),dataset.get_train().tail(30), dataset.get_train().head(20)
    df_train['labels'] = df_train['text']
    df_verify['labels'] = df_verify['text']
    df_test['labels'] = df_test['text']

    dataset.load_from_pandas(df_train,df_verify,df_test)

    sentences = df_test.text.tolist()
    
    print("Begin train model")
    ft = CLLMLLMGenerate()
    ft.load_raw(quantization_config)
    result = ft.train(dataset, max_length = 512)
    print(result)
    ft.save()
    del ft
    
    print("Begin test a fine tuned model")
    ft1 = CLLMLLMGenerate()
    ft1.load_ft(quantization_config = quantization_config)
    predicts = ft1.predict(sentences, max_new_tokens = 128)
    print(predicts)

if __name__ == "__main__":
    main()
