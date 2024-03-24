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
    def auto_fit(train_data,max_p=4,max_d=2,max_q=4):
        train_data = train_data.astype("float64")
        result_arima,min_predicted,min_p,min_q,min_d,min_error= None,None,None,None,None,np.inf
        for p in range(4):
            for q in range(4):
                for d in range(2):
                    model = sm.tsa.arima.ARIMA(train_data, order=(p, d, q))
                    res_arima = model.fit()
                    predicted = res_arima.predict()
                    error = (np.sqrt(sum((predicted-train_data).dropna()**2/train_data.size)))
                    if error < min_error:
                        result_arima = res_arima
                        min_predicted = predicted
                        min_error = error
                        min_p = p
                        min_d = d
                        min_q = q
                    
        return result_arima,min_p,min_d,min_q

    @staticmethod
    def decomposition(ds_data,model='additive',period=7):
        decomposition = seasonal_decompose(ds_data,period=period,model=model)
        return decomposition

def main():
    ds_data = pd.Series([1,2,3,4,5,6])
    transformed_data = CSHDataProcess.get_abnormal(ds_data)
    ds_data = ds_data.drop(transformed_data.index)
    ds_data = ds_data[ds_data>0]
    result_arima,min_p,min_d,min_q = CSHArima.auto_fit(ds_data,4,2,4)
    CSHArima.show_model(result_arima)
    predict_data = result_arima.predict()
    CSHArima.show_result(ds_data,predict_data)
    decomposition = CSHArima.decomposition(ds_data,period=2)
    decomposition.plot()
    plt.show()

if __name__ == "__main__":         
    main()     
