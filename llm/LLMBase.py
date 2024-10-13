#task:generate,classify,regression,embedding,qa
import torch,gc,os,random,warnings,sys
from accelerate import Accelerator
#for llama
from transformers import AutoTokenizer,LlamaTokenizer
from transformers import AutoModelForCausalLM as ModelGenerate
from transformers import AutoModelForSeq2SeqLM as ModelQA
from transformers import AutoModelForSequenceClassification as ModelClass
from transformers import LlamaModel as ModelEmbedding
#for Qwen
#from transformers import Qwen2ForCausalLM as ModelGenerate
#from transformers import AutoModelForCausalLM as ModelQA
#from transformers import Qwen2ForSequenceClassification as ModelClass
#from transformers import Qwen2Model as ModelEmbedding
#from transformers import Qwen2Tokenizer as AutoTokenizer

from transformers import Trainer,DataCollatorWithPadding,AutoModelForSequenceClassification
from LLMCommon import CLLMCommon
from LLMDataset import CLLMDataset
from datasets import DatasetDict, Dataset,load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model,PeftModel, PeftConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
from Config import quantization_config,training_config,g_model_root,g_model_base,g_default_max_length,g_default_max_new_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub")
warnings.filterwarnings("ignore")

class CLLMBase:

    def __init__(self,task="generate"):
        self.m_accelerator = Accelerator()
        self.m_task = task
        self.m_model = None
        self.m_tokenizer = None
        self.m_model_name = g_model_base
        self.m_num_labels = 0
        self.m_trainer = None

    def __del__(self):
        self.clear_gpu()    
    
    def init_tokenizer(self):
        self.m_tokenizer.pad_token_id = self.m_tokenizer.eos_token_id
        self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
        self.m_tokenizer.padding_side = 'left'

        self.m_model.config.pad_token_id = self.m_tokenizer.pad_token_id
        self.m_model.config.pad_token = self.m_tokenizer.pad_token
        self.m_model.config.bos_token_id = self.m_tokenizer.bos_token_id
        self.m_model.config.bos_token = self.m_tokenizer.bos_token
        self.m_model.config.eos_token_id = self.m_tokenizer.eos_token_id
        self.m_model.config.eos_token = self.m_tokenizer.eos_token
        self.m_model.config.use_cache = False
        self.m_model.config.pretraining_tp = 1    

    def add_tokens(self,custom_tokens):
        new_tokens = set(custom_tokens) - set(self.m_tokenizer.vocab.keys())
        self.m_tokenizer.add_tokens(list(new_tokens))
        self.m_model.resize_token_embeddings(len(self.m_tokenizer))
    
    def clear_gpu(self):
        if self.m_tokenizer:
            del self.m_tokenizer
            self.m_tokenizer = None
        if self.m_model:
            del self.m_model
            self.m_model = None
        if self.m_trainer:
            del self.m_trainer
            self.m_trainer = None
        
        gc.collect()
        if torch.cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def save(self,model_directory):
        model_name = "%s/%s"%(model_directory,self.m_task)
        os.makedirs(model_name, exist_ok=True)
        if self.m_trainer:
            self.m_trainer.save_model("%s/ft"%model_name)
            self.m_tokenizer.save_pretrained("%s/ft"%model_name)
        elif self.m_accelerator.is_local_main_process:
            unwrapped_model = self.m_accelerator.unwrap_model(self.m_model)
            unwrapped_model.save_pretrained("%s/base"%model_name)
            self.m_tokenizer.save_pretrained("%s/base"%model_name)

    #model_type: raw,local,ft
    def load(self,model_type = "raw" ,model_name_or_path=g_model_base,quantization_config = None, num_labels=0):
        
        self.clear_gpu()
        if model_type == 'raw':
            self.m_model_name = model_name_or_path
        elif model_type == 'local':
            self.m_model_name = "%s/%s/base"%(model_name_or_path,self.m_task)
        elif model_type == 'ft':
            self.m_model_name = "%s/%s/ft"%(model_name_or_path,self.m_task)
        else:
            print("Unknow model type (only raw|local|ft supported)",model_type)
            return
        
        print("Begin loadding",self.m_model_name)

        self.m_num_labels = num_labels
        tokenizer = AutoTokenizer.from_pretrained(self.m_model_name,load_in_8bit=True)
        if self.m_task == "generate":
            if quantization_config == None:
                model = ModelGenerate.from_pretrained(self.m_model_name,device_map="auto")
            else:
                model = ModelGenerate.from_pretrained(self.m_model_name,device_map="auto",quantization_config=quantization_config)
        elif self.m_task == "regression":
            if quantization_config == None:
                model = ModelClass.from_pretrained(self.m_model_name,device_map="auto",num_labels=self.m_num_labels)
            else:
                model = ModelClass.from_pretrained(self.m_model_name,device_map="auto",quantization_config=quantization_config,num_labels=self.m_num_labels)
        elif self.m_task == "classify":
            if quantization_config == None:
                model = ModelClass.from_pretrained(self.m_model_name,device_map="auto",num_labels=self.m_num_labels)
            else:
                model = ModelClass.from_pretrained(self.m_model_name,device_map="auto",quantization_config=quantization_config,num_labels=self.m_num_labels)
        elif self.m_task == "embedding":
            if quantization_config == None:
                model = ModelEmbedding.from_pretrained(self.m_model_name,device_map="auto")
            else:
                model = ModelEmbedding.from_pretrained(self.m_model_name,device_map="auto",quantization_config=quantization_config)
        elif self.m_task == "qa":
            if quantization_config == None:
                model = ModelQA.from_pretrained(self.m_model_name,device_map="auto")                                
            else:
                model = ModelQA.from_pretrained(self.m_model_name,device_map="auto",quantization_config=quantization_config)
        else:
            print("Only generate,classify,regression,embedding are supported", self.m_task)
            return
        
        self.m_model = self.m_accelerator.prepare(model)
        self.m_tokenizer = self.m_accelerator.prepare(tokenizer)
        self.init_tokenizer()
        
    def get_model(self):
        return self.m_model
    
    def get_tokenizer(self):
        return self.m_tokenizer

    def get_trainer(self):
        return self.m_trainer

    def set_trainer(self,trainer):
        self.m_trainer = trainer

    def get_input(self,input_list,max_length=g_default_max_length):
        inputs = self.m_tokenizer(input_list, return_tensors="pt",max_length=max_length,padding=True,truncation=True)
        first_device = next(self.m_model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        return inputs
        
    def generate(self,inputs, max_length = g_default_max_length, max_new_tokens = g_default_max_new_tokens):
        with torch.no_grad():
            generated_outputs = self.m_model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens,max_length=max_length,num_return_sequences=1,attention_mask=inputs["attention_mask"],pad_token_id=self.m_tokenizer.eos_token_id)
            generated_texts = [self.m_tokenizer.decode(output, skip_special_tokens=True) for output in generated_outputs]
            return generated_texts
    
    def predict(self,inputs,max_length=g_default_max_length , max_new_tokens = g_default_max_new_tokens):
        if self.m_task in ['generate','qa']:
            return self.generate(inputs,max_length,max_new_tokens)

        with torch.no_grad():
            if self.m_task == "embedding":
                outputs = self.m_model(**inputs)
                result = outputs.last_hidden_state
            elif self.m_task == "regression":
                outputs = self.m_model(**inputs)
                result = outputs['logits']
            elif self.m_task == "classify":
                outputs = self.m_model(**inputs)
                result = outputs['logits']
            elif self.m_task == "generate":          #will not executed,for debug
                input_ids = inputs['input_ids']
                generated_ids = input_ids
                for _ in range(max_length - input_ids.shape[1]):
                    outputs = self.m_model(input_ids=generated_ids)
                    logits = outputs.logits
                    next_token_logits = logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                    if next_token_id == self.m_tokenizer.eos_token_id:
                        break
                generated_text = self.m_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                result = generated_text
            elif self.m_task == 'qa':               #will not executed,for debug 
                outputs = self.m_model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                start_probs = torch.softmax(start_logits, dim=1)
                end_probs = torch.softmax(end_logits, dim=1)
                start_index = torch.argmax(start_probs)
                end_index = torch.argmax(end_probs)
                answer_ids = inputs['input_ids'][0][start_index:end_index + 1]
                result = self.m_tokenizer.decode(answer_ids, skip_special_tokens=True)
            else:
                print("Task is NOT supported (only generate,embedding,regression,classify,qa are supported)",self.m_task)

            del inputs
            return result

def main():
    
    print("demo generate")
    model1 = CLLMBase(task='generate')
    model1.load(model_type='raw')
    text = ["Sun is red, blood likes sun. blood is read. Is the statement true? Answer YES or NO and DO NOT output additional text."]

    def test(model):
        inputs = model.get_input(text, max_length = 128)
        outputs1 = model.generate(inputs, max_new_tokens = 128)
        print("output 1",outputs1)
        output2 = model.forward(inputs, max_length = 128)[len(text[0]):]
        print("output 2",output2)

    test(model1)

    print("demo save & load")
    model1.save("./models")
    del model1
    
    model2 = CLLMBase(task = 'generate')
    model2.load(model_type='local',model_name_or_path="./models")
    test(model2)

if __name__ == "__main__":
    main()
