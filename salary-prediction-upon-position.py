# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:03:51 2017

@author: zaghlollight
"""

#import lib and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#split Data 
'''
we don't need to do split data because the data here is very small

'''

#fiting linear regression and polynomial regression model to compare between result

from sklearn.linear_model import LinearRegression
linReg=LinearRegression()
linReg.fit(x,y)

#polynomial model
from sklearn.preprocessing import PolynomialFeatures
polyReg=PolynomialFeatures(degree=2)
Xpoly=polyReg.fit_transform(x)
linReg2=LinearRegression()
linReg2.fit(Xpoly,y)

#visualise linear regression 
plt.scatter(x,y,color='red')
plt.plot(x,linReg.predict(x),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()
#visualise poly regression
plt.scatter(x,y,color='red')
plt.plot(x,linReg2.predict(polyReg.fit_transform(x)),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()

#change polynomial degree to optmize the model

polyReg=PolynomialFeatures(degree=3)
Xpoly=polyReg.fit_transform(x)
linReg2=LinearRegression()
linReg2.fit(Xpoly,y)

plt.scatter(x,y,color='red')
plt.plot(x,linReg2.predict(polyReg.fit_transform(x)),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()

#increment polynomial degree to optmize the model
polyReg=PolynomialFeatures(degree=4)
Xpoly=polyReg.fit_transform(x)
linReg2=LinearRegression()
linReg2.fit(Xpoly,y)

Xgrid=np.arange(min(x),max(x),0.1)
Xgrid=Xgrid.reshape((len(Xgrid)),1)
plt.scatter(x,y,color='red')
plt.plot(Xgrid,linReg2.predict(polyReg.fit_transform(Xgrid)),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('position level')
plt.ylabel('Sallary')
plt.show()

#predicting result with linear regression 
linReg.predict(6.2)
print(linReg.intercept_,linReg.coef_)
#predicting result with polynomial regression
linReg2.predict(polyReg.fit_transform(6.2))
print(linReg2.intercept_,linReg2.coef_)
