'''
自动化训练分类模型
1.模型训练
2.模型评估
'''
import numpy as np
import pandas as pd
import pickle,h2o,json,os,sys,time,datetime,warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from SHCommon import CSHCommon
from SHSample import CSHSample
from tqdm.notebook import tqdm
from IPython.display import display
from h2o.automl import H2OAutoML
from h2o.estimators import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore") 
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)

'''
使用h2o进行automl
'''
class CSHModelRegression():
  
    def __init__(self,ip='localhost',port=54321):
        
        #h2o.init(nthreads = -1, verbose=False)
        #h2o.connect(ip=ip,port=port,verbose=False)

        self.m_models = {
            "rf":H2ORandomForestEstimator(ntrees=25,max_depth=10,sample_rate=0.5,nbins=5,min_rows=50),
            "ann":H2ODeepLearningEstimator(hidden=[100, 100]),
            "glm":H2OGeneralizedLinearEstimator(),
            "gbm":H2OGradientBoostingEstimator(),
            "xgboost":H2OXGBoostEstimator()
        }
        
    def train(self,df_sample,x_columns = [] ,y_column='y',train_ratio = 0.85):
        
        if x_columns == []:
            total_columns = df_sample.keys().tolist()
        else:
            total_columns = x_columns
            if not y_column in total_columns:
                total_columns.append(y_column)
            
        df_temp = df_sample[total_columns]
        
        df_h2o = h2o.H2OFrame(df_temp)
        if train_ratio > 0:
            df_train, df_valid = df_h2o.split_frame(ratios=[train_ratio], seed=1234)
        else:
            df_train = df_h2o
            
        x = df_train.columns
        y = y_column
        x.remove(y)
        
        for key in self.m_models:
            model = self.m_models[key]
            #print("begin train ",key)
            if train_ratio > 0:
                model.train(x=x, y=y,training_frame=df_train,validation_frame=df_valid)
            else:
                model.train(x=x, y=y,training_frame=df_train)
            #print("end train ",key)
    
    def load(self,model_path):
        for key in self.m_models:
            folder = "%s/%s"%(os.path.abspath(model_path),key)
            model = self.m_models[key]
            all_models = os.listdir(folder)
            all_models.sort()
            for file_name in all_models:
                model_file = os.path.join(folder, file_name)
                print("loading",model_file)
                self.m_models[key] = h2o.load_model(model_file)
   
    def save(self,model_path):
        for key in self.m_models:
            folder = "%s/%s"%(os.path.abspath(model_path),key)
            model = self.m_models[key]
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            for file_name in os.listdir(folder):
                model_file = os.path.join(folder, file_name)
                os.remove(model_file)
            
            model_saved = h2o.save_model(model=model, path=folder, force=True)
            print("save model to ",model_saved)

    def predict(self,df_sample,x_columns = [],y_column='y'):
        
        if x_columns == []:
            total_columns = df_sample.keys().tolist()
        else:
            total_columns = x_columns
         
        if y_column in total_columns:
            total_columns.remove(y_column)
                
        if y_column in df_sample.keys().tolist():
            y_true = df_sample[y_column].tolist()
        else:
            y_true = None
            
        df_temp = df_sample[total_columns]
        df_h2o = h2o.H2OFrame(df_temp)

        result = []
        for key in self.m_models:
            model = self.m_models[key]
            training_columns = model._model_json['output']['names'][:-1]
            for col in training_columns:
                if col not in df_temp.columns:
                    print("NO training column %s found not"%col)
                    df_temp = df_temp.reindex(columns=training_columns, fill_value=0)  
                    df_h2o = h2o.H2OFrame(df_temp)
                    break
            df_t = model.predict(df_h2o).as_data_frame()
            df_t['model'] = key
            df_t['true'] = y_true
            result.extend(json.loads(df_t.to_json(orient='records')))
            
        return pd.DataFrame(result)
    
    def evaluate(self,df_sample,x_columns = [],y_column='y'):
        df_pred = self.predict( df_sample,x_columns=x_columns , y_column=y_column )
        result = []
        for model,df in df_pred.groupby('model'):
            y_pred = df['predict'].tolist()
            y_true = df['true'].tolist()
            tmp = {}
            tmp['model'] = model
            tmp['mse'] = mean_squared_error(y_true,y_pred)
            tmp['rmse'] = np.sqrt(tmp['mse'])
            tmp['mae'] = mean_absolute_error(y_true,y_pred)
            tmp['r2'] = r2_score(y_true,y_pred)
            result.append(tmp)

        return pd.DataFrame( result )

    def importance(self):
        result = []
        for key in self.m_models:
            model = self.m_models[key]
            df_temp = model.varimp(use_pandas=True)
            if type(df_temp) == pd.DataFrame:
                df_temp['model'] = key
                result.extend(json.loads(df_temp.to_json(orient='records')))
        df_importance = pd.DataFrame(result).reset_index(drop=True)
        return df_importance

def main():

    h2o.init(nthreads = -1, verbose=False)

    df_sample = CSHSample.get_random_regression(1000)
    df_sample['y'] = df_sample['y'].astype(float)
    df_train,df_test = CSHSample.split_dataset(df_sample)
    print(df_train)

    model_1 = CSHModelRegression()
    model_1.train(df_train,y_column='y')

    df_pred = model_1.predict(df_sample,y_column='y')
    display(df_pred)

    df_verify = model_1.evaluate(df_sample,y_column='y')
    display(df_verify)

    df_importance = model_1.importance()
    display(df_importance)

if __name__ == "__main__":
    main()
