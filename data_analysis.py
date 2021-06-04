# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:57:22 2019

@author: milk
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
#2
iris=load_iris()
print(iris.DESCR)
iris.keys()
print(len(iris.feature_names))
print(iris.feature_names)
print(len(iris.data))
print(len(iris.target))
a=iris.data[:,2:]
b=iris.target
plt.scatter(a[:,0],a[:,1],c=b,cmap=plt.cm.Paired)
plt.xlabel('patal length')
plt.ylabel('petal width')
#3
x = pd.DataFrame(iris.data,columns = iris.feature_names)
y = pd.DataFrame(iris.target,columns=['target'])
data = pd.concat([x,y],axis=1) 
data_x = x.values
data_y = y.values
validation_size=1/3
seed=3
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=validation_size, random_state=seed)
model = GaussianNB()
model.fit(x_train,y_train)
predict=model.predict(x_test)
print(model.predict(x_test))
acc=0
for i in range(len(y_test)):
    if y_test[i]==predict[i]:
        acc=acc+1
    else:
        acc=acc
ACC=acc/len(y_test)
print('accuracy=',ACC)