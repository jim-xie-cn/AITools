#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env bash
from sklearn.cluster import KMeans,AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,MeanShift,OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from SHSample import CSHSample
import pandas as pd
import numpy as np

class CSHCluster:
    def __init__(self,cluster_count):
        self.m_models = {
            "KMeans": KMeans(n_clusters=cluster_count, random_state=False),
            # "AffinityPropagation":AffinityPropagation(damping=0.9),
            "AgglomerativeClustering": AgglomerativeClustering(n_clusters=cluster_count),
            "Birch": Birch(threshold=0.01, n_clusters=cluster_count),
            #"DBSCAN": DBSCAN(eps=0.30, min_samples=9),
            # "MeanShift":MeanShift(),
            # "OPTICS":OPTICS(eps=0.30, min_samples=9),
            "SpectralClustering": SpectralClustering(n_clusters=cluster_count, random_state=False),
            "GaussianMixture": GaussianMixture(n_components=cluster_count, covariance_type='full', random_state=False)
        }

    def fit_predict(self,df_sample):
        X_train = df_sample.to_numpy()
        cluster_result = []
        for model_name in self.m_models:
            model = self.m_models[model_name]
            yhat = model.fit_predict(X_train)
            clusters = np.unique(yhat)
            for cluster in clusters:
                row_ix = np.where(yhat == cluster)
                tmp = {}
                tmp['model'] = model_name
                tmp['cluster'] = cluster
                tmp['rows'] = row_ix
                cluster_result.append(tmp)
        return pd.DataFrame(cluster_result)

    def evaluate(self,df_sample):
        X_train = df_sample.to_numpy()
        result = []
        for model_name in self.m_models.keys():
            model = self.m_models[model_name]
            tmp = {"model_name": model_name}
            yhat = model.fit_predict(X_train)
            tmp["silhouette_score"] = silhouette_score(X_train, yhat)
            tmp["calinski_harabasz_score"] = calinski_harabasz_score(X_train, yhat)
            tmp["davies_bouldin_score"] = davies_bouldin_score(X_train, yhat)
            result.append(tmp)
        return pd.DataFrame(result)

    def sample_cluster(self,df_cluster_result,df_sample,model_name):
        df_tmp = df_cluster_result[df_cluster_result['model'] == model_name]
        df_ret = df_sample.copy(deep = True)
        df_ret['_cluster'] = ""
        for rows, cluster in zip(df_tmp['rows'], df_tmp['cluster']):
            indexes = rows[0].astype(int)
            df_ret.iloc[indexes, -1] = cluster
        return df_ret

def main():
    df_sample = CSHSample.get_random_cluster()
    test = CSHCluster(2)
    df_result = test.fit_predict(df_sample)
    print(df_result)
    df_perf = test.evaluate(df_sample)
    print(df_perf)
    df_cluster_sample = test.sample_cluster(df_result,df_sample,"KMeans")
    print(df_cluster_sample)
if __name__ == "__main__":
    main()



