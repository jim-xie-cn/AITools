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
    def auto_fit(ds_train,
                 ds_test,
                 seasonal_order_range=((1,2),(1,2),(1,10),(1,7))):
    
        ds_train = ds_train.astype("float64")
        ds_test = ds_test.astype("float64")
        order_p = seasonal_order_range[0]
        order_d = seasonal_order_range[1]
        order_q = seasonal_order_range[2]
        order_period = seasonal_order_range[3]
        ret_order = None
        min_error = np.inf
        
        for p in range(order_p[0],order_p[1]):
            for d in range(order_d[0],order_d[1]):
                for q in range(order_q[0],order_q[1]):
                    for period in range(order_period[0],order_period[1]):
                        model = sm.tsa.arima.ARIMA(ds_train, seasonal_order=(p, d, q,period))
                        res_arima = model.fit()
                        predicted = res_arima.forecast(ds_test.shape[0])
                        error = (np.sqrt(sum((predicted-ds_test).dropna()**2/ds_test.size)))
                        if error < min_error:
                            result_arima = res_arima
                            min_predicted = predicted
                            min_error = error
                            ret_order = (p,d,q,period)
                    
        return result_arima,ret_order

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
    model,ret_order = CSHArima.auto_fit(ds_train = ds_data.head(4),
                                        ds_test = ds_data.tail(2),
                                        seasonal_order_range = seasonal_order_range)

    CSHArima.show_model(model)
    predict_data = model.predict()
    CSHArima.show_result(ds_data,predict_data)
    decomposition = CSHArima.decomposition(ds_data,period=2)
    decomposition.plot()
    plt.show()

if __name__ == "__main__":         
    main()     
