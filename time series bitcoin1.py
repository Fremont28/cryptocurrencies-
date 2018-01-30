import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot

#bitcoin pricing data 1/18/18 
def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


#get the kraken btc price exchange data
#historical bitcoin exhange rate
btc_usd_price=get_quandl_data('BCHARTS/KRAKENUSD')
btc_usd_price.head(5)
type(btc_usd_price) #type pandas df

#convert into a tiem series object 
ts_obj=btc_usd_price['Weighted Price']
type(ts_obj) #pandas series 
#plot series
ts_obj.plot()
pyplot.show() #data is not stationary 
#ACF 
autocorrelation_plot(ts_obj)
pyplot.show()

#fit the model (significant if outside the lines -1 to 1 correlations)
model=ARIMA(ts_obj,order=(20,1,0))
model_fit=model.fit(disp=0)
model_fit.summary()
#plot residuals 
residuals=DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

residuals.plot(kind='kde')
pyplot.show() #residual errors centered around zero 

residuals.describe() #mean centered around zero 

###rolling ARIMA forecast 
#usa forecast()-one-step forecast using the model 
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot 
from statsmodels.tsa.arima_model import ARIMA 
from sklearn.metrics import mean_squared_error

X=ts_obj.values 
size=int(len(X)*0.66)
train,test=X[0:size],X[size:len(X)] #test, train sets
historical=[i for i in train] #class list
predictions=list()
for t in range(len(test)):
    model=ARIMA(historical,order=(7,1,0))
    model_fit=model.fit(disp=0)
    output=model_fit.forecast()
    yhat=output[0]
    predictions.append(yhat)
    obs=test[t]
    historical.append(obs)
    print(yhat,obs) #predicted, observed
error=mean_squared_error(test,predictions)
error #MSE 
#plot predictions vs. acutal values
pyplot.plot(test)
pyplot.plot(predictions,color="orange")
pyplot.show()

