# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:35:06 2017

@author: zaghlollight
"""
#import lib and data set
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#feat. scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)




#fiting SVR to the data set
from sklearn.svm import SVR
reg=SVR(kernel='rbf')
reg.fit(x,y)

#predecting a new result
y_pred=reg.predict(sc_x.transform(np.array([[6.5]])))
print(sc_y.inverse_transform(y_pred))

#vis
plt.scatter(x,y,color='red')
plt.plot(x,reg.predict(x),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()