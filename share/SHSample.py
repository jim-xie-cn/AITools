import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
from imblearn.over_sampling import SMOTE
from IPython.display import display
from warnings import filterwarnings
filterwarnings("ignore") 
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)

class CSHSample():
    
    @staticmethod 
    def get_random_regression(n_sample = 100):
        df = pd.DataFrame()
        features, target = make_regression(n_samples=n_sample,
                                           n_features=2,
                                           n_targets=1,
                                           random_state=100,
                                           noise = 1)
        df['x1'] = features[:,0]
        df['x2'] = features[:,1]
        df['y'] = features[:,0] + features[:,1]
        return df
    
    @staticmethod 
    def get_random_classification(n_sample = 100 , n_feature =2, n_class = 2):
        df = pd.DataFrame()
        X, y = make_classification(n_samples=n_sample,
                                   n_features=n_feature,
                                   n_redundant=0,
                                   n_classes=n_class, # binary target/label
                                   n_clusters_per_class = 1,
                                   n_informative = n_feature,
                                   flip_y=0.1,  #high value to add more noise
                                   random_state=100)
        for i in range(n_feature):
            df['x%d'%i] = X[:,i]
        
        df['y'] = y
        return df

    @staticmethod
    def get_random_cluster(n_sample = 100):
        centers = [(-5, -5), (5, 5)]
        cluster_std = [0.8, 1]
        X, y = make_blobs(n_samples=n_sample , cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
        df = pd.DataFrame()
        df['x1'] = X[:,0]
        df['x2'] = X[:,1]
        df['y'] = y
        return df

    @staticmethod
    def split_dataset(df,test_ratio = 0.2):
        df_train,df_test = train_test_split(df, test_size = test_ratio)
        return df_train.reset_index(drop=True),df_test.reset_index(drop=True)

    @staticmethod
    def resample_smote(df_sample,x_columns=[],y_column='label',random_state=42):

        if x_columns == []:
            total_columns = []
            for col in df_sample.keys().tolist():
                if col != y_column:
                    total_columns.append(col)
        else:
            total_columns = x_columns

        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(df_sample[total_columns], df_sample[y_column])
        df_ret = X_resampled
        df_ret[y_column] = y_resampled
        return df_ret
    
    @staticmethod
    def resample_balance(df_data,y_column = 'labels'):
        df_minority = df_data[df_data[y_column] == df_data[y_column].value_counts().idxmin()]
        df_majority = df_data[df_data[y_column] == df_data[y_column].value_counts().idxmax()]
        df_minority_upsampled = df_minority.sample(len(df_majority), replace=True)
        df_balanced = pd.concat([df_minority_upsampled, df_majority])
        df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
        return df_balanced
        
def main():
    df_classification = CSHSample.get_random_classification(1000,n_feature=10,n_class=4)
    df_regression = CSHSample.get_random_regression()
    df_cluster = CSHSample.get_random_cluster()
    display(df_classification)
    display(df_regression)
    display(df_cluster)

    df_train,df_test = CSHSample.split_dataset(df_regression)
    display(df_train.shape)
    display(df_test.shape)

    df_sample = CSHSample.resample_smote(df_classification,y_column='y')
    df_sample = CSHSample.resample_balance(df_classification,y_column='y')

    display(df_sample)

if __name__ == "__main__":
    main()

