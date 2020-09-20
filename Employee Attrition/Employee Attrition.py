#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd


# In[15]:


df = pd.read_csv("C:/Users/Dipanjan/Desktop/Python/Project/Employee Attrition/1.csv")


# In[16]:


#categorical variables
columns = ['BusinessTravel' , 'Department' , 'EducationField' , 'Gender' , 'JobRole' , 'MaritalStatus' , 'Over18' , 'OverTime']


# In[17]:


#function to apply onehot-encoder
def category_onehot_multcols(multcolumns,final_df):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        #print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[18]:


#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(df['Attrition'])

X=df.copy()
X.drop(['Attrition'] , axis=1 , inplace = True)
X = category_onehot_multcols(columns,X)


# In[19]:


#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2)


# In[20]:


#implementing model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train , Y_train)
Y_pred = model.predict(X_test )


# In[21]:


#performance
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
cm=confusion_matrix(Y_test,Y_pred)
accuracy_score=accuracy_score(Y_test,Y_pred)
precision_score=precision_score(Y_test,Y_pred)
recall_score=recall_score(Y_test,Y_pred)
print (f'Confusion matrix: {cm}')
print (f'Accuracy score: {accuracy_score}')
print (f'Precision score: {precision_score}')
print (f'Recall score: {recall_score}')

