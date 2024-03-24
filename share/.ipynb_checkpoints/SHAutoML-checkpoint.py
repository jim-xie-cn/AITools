import h2o
import time,json,os,sys
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ORandomForestEstimator

g_models = {
    "bayes":H2ONaiveBayesEstimator(),
    "glm":H2OGeneralizedLinearEstimator(nfolds = 4),
    "rf":H2ORandomForestEstimator(nfolds = 4),
    "gbm":H2OGradientBoostingEstimator(nfolds=4),
    #"svm":H2OSupportVectorMachineEstimator(),
    "xgboost":H2OXGBoostEstimator(nfolds=4),
    "deeplearn":H2ODeepLearningEstimator(hidden=[200, 200],nfolds = 4)
}

def get_performance(model,df_test):
    result = {}
    perf = model.model_performance(df_test)
    result['mcc'] = json.loads(str(perf.mcc()))
    result['f1'] = json.loads(str(perf.F1()))
    result['f05'] = json.loads(str(perf.F0point5()))
    result['f2'] = json.loads(str(perf.F2())) 
    result['accuracy'] =json.loads(str(perf.accuracy())) 
    result['logloss'] = json.loads(str(perf.logloss())) 
    result['recall'] = json.loads(str(perf.recall())) 
    result['precision'] = json.loads(str(perf.precision())) 
    result['gini'] = perf.gini()
    result['auc'] = perf.auc()
    result['aucpr'] = perf.aucpr()
    result['roc'] = json.loads(json.dumps(perf.roc()))
    result['fpr'] = json.loads(json.dumps(perf.fpr()))
    result['tpr'] = json.loads(json.dumps(perf.tpr()))
    result['confusion_matrix'] = perf.confusion_matrix().to_list()
    return result

def train( sample_file , sample_name ):

    h2o_prepare()

    #load sample from file
    df = pd.read_csv(sample_file,index_col=0)
    df_all = preprocess(df)
    if not is_sample_qualified(df_all):
        print("sample data is not qualified. at least, samples contain two type lables (0 and 1) and sample count more than 100")
        return None

    #prepare train,valid and test sample and check sample qulity    
    df_h2o = h2o.H2OFrame(df_all)
    df_train, df_valid = df_h2o.split_frame(ratios=[0.85], seed=1234)
    if not is_sample_qualified(df_train):
        print("train sample is not qualified. at least, samples contain two type lables (0 and 1) and sample count more than 100")
        return None

    if not is_sample_qualified(df_valid):
        print("valid sample is not qualified. at least, samples contain two type lables (0 and 1) and sample count more than 100")
        return None

    #alloc path to save model,performane metric and train sample set.   
    train_root_path = "%s%s/"%( g_model_path , sample_name )
    
    if not train_root_path or train_root_path == "/" or not sample_name :
        print("error train_root_path ",train_root_path)
        return None

    os.system("rm -rf %s"%train_root_path)
    os.system("mkdir -p %s"%train_root_path)

    df_train.as_data_frame().to_csv("%s/train.csv"%train_root_path)
    df_valid.as_data_frame().to_csv("%s/valid.csv"%train_root_path)

    x = df_train.columns
    y = "label"
    x.remove(y)
    df_train[y] = df_train[y].asfactor()
    df_valid[y] = df_valid[y].asfactor()
    print("Number of train, valid and test set : ", df_train.shape[0], df_valid.shape[0] )
    
    #Train and save customized models
    meta_info = {}
    meta_info['sample_name'] = sample_name
    for key in g_models:
        model = g_models[key]
        print("begin train ",key)
        model.train(x=x, y=y,training_frame=df_train,validation_frame=df_valid)
        #model.train(x=x, y=y,training_frame=df_train,validation_frame=df_valid,max_runtime_secs=10)

        model_file = h2o.save_model(model=model, path="%s/%s"%(train_root_path,key), force=True)
        meta_info[key] = {}
        meta_info[key]['performance'] = get_performance(model,df_valid)
        meta_info[key]['file'] = model_file
        print("end train ",key)
    with open("%s/meta.info"%train_root_path,"w") as fp:
        data = json.dumps(meta_info,indent=4)
        fp.write(data)
    print("train finished")


def main():
    h2o.init(ip="localhost",port=54321)

if __name__ == "__main__":
    main()
