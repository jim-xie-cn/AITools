import gc,os,random,torch,warnings
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,mean_squared_error,balanced_accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix,mean_squared_error,mean_absolute_error,r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
warnings.filterwarnings("ignore")
from transformers import BitsAndBytesConfig,TrainingArguments,TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from Config import quantization_config,training_config,g_model_root,g_data_root,g_default_max_length,g_model_base

class CLLMCommon(object):
    @staticmethod
    def init_random():
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def pooling_data(arr,dest_shape):
        embedding = np.array(arr)
        embedding = torch.tensor(embedding)
        pooled_tensor = F.adaptive_avg_pool2d(embedding.unsqueeze(0).unsqueeze(0), dest_shape)
        return pooled_tensor.squeeze().numpy()
        
    @staticmethod
    def evaluate_single_classify(labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        return {'accuracy': accuracy,'precision': precision,'recall': recall,'f1': f1}

    @staticmethod
    def evaluate_mul_classify(labels, preds):
        results = {}
        accuracies, precisions, recalls, f1s = [],[],[],[]
        num_classes = labels.shape[1]  # number of classes
        for i in range(num_classes):
            class_labels = labels[:, i]  # Get true labels for class i
            class_preds = preds[:, i]    # Get predicted labels for class i
            accuracy = accuracy_score(class_labels, class_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(class_labels, class_preds, average='binary')
            # Store results for this class
            results[f'class_{i}_accuracy'] = accuracy
            results[f'class_{i}_precision'] = precision
            results[f'class_{i}_recall'] = recall
            results[f'class_{i}_f1'] = f1

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        # Calculate overall metrics (macro-average across classes)
        results['macro_accuracy'] = sum(accuracies) / num_classes
        results['macro_precision'] = sum(precisions) / num_classes
        results['macro_recall'] = sum(recalls) / num_classes
        results['macro_f1'] = sum(f1s) / num_classes
        return results
   
    @staticmethod
    def evalulate_regression(labels, preds):
        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        single_squared_errors = ((preds - labels).flatten()**2).tolist()
        accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
        return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


class CustomTensorBoardCallback(TrainerCallback):

    def __init__(self):
        log_dir = "%s/logs"%g_model_root
        self.train_writer = SummaryWriter(log_dir=f"{log_dir}/train")
        self.eval_writer = SummaryWriter(log_dir=f"{log_dir}/eval")

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        #if "loss" in logs:
        #    self.train_writer.add_scalar("train/loss", logs["loss"], state.global_step)
        #if "eval_loss" in logs:
        #    self.eval_writer.add_scalar("eval/loss", logs["eval_loss"], state.global_step)
        for key in logs:
            if key.find("eval") >= 0:
                self.eval_writer.add_scalar("eval/%s"%key, logs[key], state.global_step)
            else:
                self.train_writer.add_scalar("train/%s"%key, logs[key], state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.train_writer.close()
        self.eval_writer.close()

