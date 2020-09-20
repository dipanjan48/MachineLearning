#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd


# In[43]:


df = pd.read_csv("C:/Users/Dipanjan/Desktop/Python/Project/Wine testing/winequality-white.csv")


# In[44]:


#df.dtypes


# In[45]:


X = df.copy()
X.drop(['quality'], axis=1, inplace=True)
X.drop(['density'], axis=1, inplace=True)
X.drop(['pH'], axis=1, inplace=True)
Y = df['quality']


# In[46]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(X_test.shape)


# In[47]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=18)
model.fit(X_train , Y_train)
model.score(X_test , Y_test)


# In[48]:


#finding elbow for KNN_manually


# In[49]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


#to store rmse values for different k
rmse_val = []
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, Y_train)  #fit the model
    Y_pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(Y_test,Y_pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[51]:


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()


# In[52]:


#finding elbow for KNN_method


# In[53]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,Y_train)
model.best_params_


# In[ ]:




