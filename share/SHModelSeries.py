'''
ARIMA处理类
'''
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from SHDataProcess import CSHDataProcess
import numpy as np
from IPython.display import display
from tqdm.notebook import tqdm
from itertools import product

class CSHArima:
    
    def __init__(self):
        pass
        
    @staticmethod
    def fit(dataseries,p=1,d=0,q=1):
        train_data = dataseries
        train_data = train_data.astype("float64")
        return sm.tsa.arima.ARIMA(train_data, order=(p, d, q)).fit()

    '''
    显示模型信息，关注以下指标
    AIC，越大越好
    BIC，越大越好
    P>|z|，越小越好
    Prob(Q)
    '''
    @staticmethod
    def show_model(result_arima):
        display(result_arima.summary())
    
    @staticmethod
    def predict(result_arima):
        return result_arima.predict()
        
    @staticmethod
    def forecast(result_arima,num_future_points=1):
        return result_arima.forecast(steps=num_future_points)
        
    @staticmethod
    def show_result(train_data,predicted):
        plt.figure(facecolor='white')
        error = (np.sqrt(sum((predicted-train_data).dropna()**2/train_data.size)))
        predicted.plot(color='blue', label='Predict')
        train_data.plot(color='red', label='Original')
        plt.legend(loc='best')
        plt.title('RMSE: %.4f'% error)
        plt.show()
    
    @staticmethod
    def auto_fit(ds_sample,n_verify,n_forcast, seasonal_order_range=((1,2),(1,2),(1,3),(5,7))):
        def train_once(ds_data,n_verify,n_forcast,p, d, q,period):
            model = sm.tsa.arima.ARIMA(ds_data, seasonal_order=(p, d, q ,period )).fit()
            start_pos = ds_sample.shape[0] - n_verify
            end_pos = ds_sample.shape[0]
            ds_verify = ds_sample.tail(n_verify)
            ds_predicted = model.predict(start_pos,end_pos,dynamic=True, typ='levels')
            ds_forcast = model.forecast(n_forcast)
            error = (np.sqrt(sum((ds_predicted-ds_verify).dropna()**2/ds_verify.size)))
            return model,error,ds_forcast,ds_predicted,ds_verify
            
        ds_sample = ds_sample.astype("float64")
        order_p = seasonal_order_range[0]
        order_d = seasonal_order_range[1]
        order_q = seasonal_order_range[2]
        order_period = seasonal_order_range[3]
        all_combinations = list(product(order_p, order_d, order_q, order_period))
        min_error = np.inf
        ret = {}
        for p,d,q,period in tqdm(all_combinations):
            try :
                model,error,ds_forcast,ds_predicted,ds_verify = train_once(ds_sample,n_verify,n_forcast,p,d,q,period)
                if error < min_error:
                    min_error = error
                    ret['model'] = model
                    ret['error'] = error
                    ret['paramter'] = (p,d,q,period)
                    ret['forcast'] = ds_forcast
                    ret['predict'] = ds_predicted
                    ret['verify'] = ds_verify
            except:
                print("error",p,d,q,period)
        return ret
        
    @staticmethod
    def decomposition(ds_data,model='additive',period=7):
        decomposition = seasonal_decompose(ds_data,period=period,model=model)
        return decomposition

def main():
    ds_data = pd.Series([1,2,3,4,5,6])
    transformed_data = CSHDataProcess.get_abnormal(ds_data)
    ds_data = ds_data.drop(transformed_data.index)
    ds_data = ds_data[ds_data>0]
    seasonal_order_range = ((1,4),(1,4),(3,10),(5,10))
    result = CSHArima.auto_fit(ds_data,
                                        n_verify = 2,
                                        n_forcast = 3,
                                        seasonal_order_range = seasonal_order_range)
    
    CSHArima.show_model(result['model'])
    CSHArima.show_result(result['predict'],result['verify'])
    
    decomposition = CSHArima.decomposition(ds_data,period=2)
    decomposition.plot()
    plt.show()

if __name__ == "__main__":         
    main()     
