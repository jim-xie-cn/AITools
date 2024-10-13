import os,json,sys,logging
from tqdm import tqdm
from transformers import BitsAndBytesConfig,TrainingArguments,TrainerCallback
from peft import LoraConfig
import os,torch
import numpy as np
sys.path.append("./share")
sys.path.append("./common")
g_data_root = "/data/edu/data/"
g_model_root = "/data/edu/models"
g_model_base = "meta-llama/Meta-Llama-3-8B"
g_default_batch_size = 32
g_default_max_length = 32*1024
g_default_max_new_tokens = 1024

quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                         bnb_4bit_quant_type = 'nf4',
                                         bnb_4bit_use_double_quant = False,
                                         bnb_4bit_compute_dtype = torch.bfloat16,
                                         load_in_8bit_fp32_cpu_offload=True)

lora_config_classify = LoraConfig(r=64,
                                  lora_alpha=32,
                                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj','gate_proj','down_proj','up_proj'],
                                  lora_dropout=0.05,
                                  bias='none',
                                  task_type='SEQ_CLS')

lora_config_generate = LoraConfig(r=64,
                                  lora_alpha=32,
                                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj','gate_proj','down_proj','up_proj'],
                                  lora_dropout=0.05,
                                  bias='none',
                                  task_type='CAUSAL_LM')

training_config = TrainingArguments(output_dir = '%s/checkpoint'%g_model_root,
                                    learning_rate = 0.0005,
                                    per_device_train_batch_size = 4,
                                    per_device_eval_batch_size = 2,
                                    num_train_epochs = 10,
                                    logging_steps=1,
                                    weight_decay = 0.01,
                                    evaluation_strategy = 'epoch',
                                    save_strategy = 'epoch',
                                    #fsdp = "full_shard offload",
                                    #gradient_accumulation_steps=8,
                                    #fp16=True,
                                    #save_steps=10,
                                    #save_total_limit=2,
                                    #evaluation_strategy="steps",
                                    #eval_steps=10,
                                    load_best_model_at_end=True,
                                    report_to="none")
                                    #training_config.fsdp = "offload",
                                    #training_config.fp16 = True)
                                                                   
