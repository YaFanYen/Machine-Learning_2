# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:07:44 2019

@author: milk
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#1
data=pd.read_csv("C:\\Users\\milk3\\Documents\\Machine Learning\\HW2\\tennis.csv")
print(data)
print(data.columns)
data_total=len(data)
humidity_total=data.groupby(['humidity']).size()
humidity_count=data.groupby(['humidity','play']).size()
hh=humidity_total['high']
high_humidity=hh/data_total
hp=humidity_count['high','yes']
play=hp/data_total
prob=play/high_humidity
print('probability_high humidity_play=',prob)
#2
windy_total=data.groupby(['windy']).size()
windy_count=data.groupby(['windy','play']).size()
wf=windy_total[False]
nowind=wf/data_total
wp=windy_count[False,'yes']
playy=wp/data_total
probb=playy/nowind
print('probability_no wind_play=',probb)



'''
X=pd.get_dummies(data[['outlook','temp','humidity','wind']])
Y=pd.DataFrame(data['play'])
print(X.head())
print(Y.head())

model=GaussianNB()
model.fit(X,Y)
predicted=model.predict([[False,1,0,0,0,1,0,1,0]])
print(predicted)
'''