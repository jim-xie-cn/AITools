import torch
from accelerate import Accelerator
import gc,os,random,warnings,sys
sys.path.append("./llama/")
from LLMCommon import CLLMCommon
from LLMDataset import CLLMDataset
from LLMBase import CLLMBase
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import gc,torch,os,warnings,random
from tqdm import tqdm
import pandas as pd
import numpy as np
from Config import quantization_config,g_model_root,g_data_root,g_model_base,g_default_max_length,g_default_batch_size
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub")
warnings.filterwarnings("ignore")

class CLLMModelEmbedding(object):

    def __init__(self):
        self.m_num_labels = 0
        self.m_model = CLLMBase(task='embedding')

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

    def predict(self,sentences, batch_size = g_default_batch_size, max_length=g_default_max_length):
        max_records_per_df = batch_size
        num_splits = int(np.ceil(len(sentences) / max_records_per_df))
        sub_dfs = np.array_split(sentences, num_splits)
        result = []
        for sub_sentence in tqdm(sub_dfs,desc="predicting"):
            sub_sentence = sub_sentence.tolist()
            inputs = self.m_model.get_input(sub_sentence,max_length=max_length)
            outputs = self.m_model.predict(inputs,max_length = max_length)
            pred = outputs.squeeze().cpu().numpy()
            result.extend(list(pred))
        return np.array(result)

def main():
    CLLMCommon.init_random()
    model = CLLMModelEmbedding()
    model.load_ft(quantization_config = quantization_config)
    data = ["i'm a teacher","you are a robot"]
    embedding = model.predict(data)
    print(embedding)
    print(embedding.shape)

if __name__ == "__main__":
    main()

