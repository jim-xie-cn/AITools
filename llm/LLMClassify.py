#task:generate,classify,regression,embedding,qa
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification, LlamaTokenizer, LlamaModel
import gc,os,random,warnings,sys
sys.path.append("./llama/")
from tqdm import tqdm
from LLMCommon import CLLMCommon,CustomTensorBoardCallback
from LLMDataset import CLLMDataset
from LLMBase import CLLMBase
from datasets import DatasetDict, Dataset,load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model,PeftModel, PeftConfig
from transformers import Trainer,DataCollatorWithPadding,AutoModelForSequenceClassification
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import gc,torch,os,warnings,random
import pandas as pd
import numpy as np
from Config import quantization_config,lora_config_classify,training_config,g_model_root,g_data_root,g_default_max_length,g_model_base,g_default_batch_size
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub")
warnings.filterwarnings("ignore")

class CustomClassifyTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        #loss = F.cross_entropy(logits, labels)
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)                                                   
        else:
            loss = F.cross_entropy(logits, labels)
        #return loss
        return (loss, outputs) if return_outputs else loss

class CLLMClassify(object):

    def __init__(self,num_labels):
        self.m_model = CLLMBase(task='classify')
        self.m_num_labels = num_labels

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

    def load_ft(self,model_directory = None, quantization_config = None):
        if model_directory == None:
            model_directory = "%s"%g_model_root
        self.m_model.load(model_type = "ft", model_name_or_path = model_directory, quantization_config = quantization_config, num_labels = self.m_num_labels)
        custom_tokens = self.get_custom_tokens()
        if custom_tokens:
            self.m_model.add_tokens(custom_tokens)

    def train(self,llm_dataset, max_length = g_default_max_length):
        tokenizer = self.m_model.get_tokenizer()
        model = self.m_model.get_model()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config_classify)

        tokenized_data,collate_fn=llm_dataset.get_single_classify_tokenized_data(tokenizer,max_length=max_length)

        def compute_metrics(evaluations):
            preds, labels = evaluations
            preds = np.argmax(preds, axis=1)
            return CLLMCommon.evaluate_single_classify(preds, labels)

        ds_label = pd.Series(llm_dataset.get_dataset()['train']['labels'])
        class_weights=(1/ds_label.value_counts(normalize=True).sort_index()).tolist()
        class_weights=torch.tensor(class_weights)
        class_weights=class_weights/class_weights.sum()

        trainer = CustomClassifyTrainer(model = model,
                                        args = training_config,
                                        train_dataset = tokenized_data['train'],
                                        eval_dataset = tokenized_data['val'],
                                        tokenizer = tokenizer,
                                        data_collator = collate_fn,
                                        compute_metrics = compute_metrics,
                                        class_weights=class_weights,
                                        callbacks=[CustomTensorBoardCallback()])
        
        self.m_model.set_trainer(trainer)

        return self.m_model.get_trainer().train()

    def predict(self,sentences, batch_size = g_default_batch_size, max_length=g_default_max_length):
        max_records_per_df = batch_size
        num_splits = int(np.ceil(len(sentences) / max_records_per_df))
        sub_dfs = np.array_split(sentences, num_splits)
        result = []
        probability = []
        for sub_sentence in tqdm(sub_dfs,desc="predicting"):
            sub_sentence = sub_sentence.tolist()
            inputs = self.m_model.get_input(sub_sentence,max_length=max_length)
            outputs = self.m_model.predict(inputs,max_length = max_length)
            pred = torch.argmax(outputs, dim=-1).cpu().numpy().tolist()
            result.extend(pred)
            probability.extend(outputs.cpu().numpy().tolist())
        return result,probability

def main():
    CLLMCommon.init_random()
    #prepare data
    dataset = CLLMDataset()
    dataset.load_from_huggingface("ag_news")
    df_train,df_verify,df_test = dataset.get_train().sample(200),dataset.get_train().tail(100), dataset.get_test().head(50)
    df_train = df_train.rename(columns={'label': 'labels'})
    df_verify = df_verify.rename(columns={'label': 'labels'})
    df_test = df_test.rename(columns={'label': 'labels'})
    dataset.load_from_pandas(df_train,df_verify,df_test)
    
    sentences = df_test.tail(100).text.tolist()
    labels = df_test.tail(100).labels.tolist()
    
    #print("Begin train a model")
    #ft = CLLMClassify(4)
    #ft.load_raw( quantization_config = quantization_config )
    #result = ft.train(dataset,max_length=512)
    #print(result)
    #ft.save()
    #del ft

    print("Begin test a model")
    ft1 = CLLMClassify(4)
    ft1.load_ft(quantization_config = quantization_config)
    predicts,probability = ft1.predict(sentences)
    print(labels,predicts)
    result = CLLMCommon.evaluate_single_classify(labels, predicts)
    print(result)

if __name__ == "__main__":
    main()

