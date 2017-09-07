# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:26:01 2017

@author: zaghlollight
"""

#import lib and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#train random forest regression
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=10,random_state=0)
reg.fit(x,y)
#predict new salary 
y_pred=reg.predict(6.5)

#visualise the model
Xgrid=np.arange(min(x),max(x),0.1)
Xgrid=Xgrid.reshape((len(Xgrid)),1)
plt.scatter(x,y,color='red')
plt.plot(Xgrid,reg.predict(Xgrid),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()

#multiply number of estimator
reg=RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(x,y)

#multiply number of estimator
reg=RandomForestRegressor(n_estimators=300,random_state=0)
reg.fit(x,y)