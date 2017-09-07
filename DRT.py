# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 00:27:20 2017

@author: zaghlollight
"""

#import lib and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#train model using DTR
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0)
reg.fit(x,y)

#predicting a new result
y_pred=reg.predict(6.5)


#visualising the decision tree regression result
plt.scatter(x,y,color='red')
plt.plot(x,reg.predict(x),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()

Xgrid=np.arange(min(x),max(x),0.1)
Xgrid=Xgrid.reshape((len(Xgrid)),1)
plt.scatter(x,y,color='red')
plt.plot(Xgrid,reg.predict(Xgrid),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()
