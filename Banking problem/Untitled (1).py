#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
import pandas as pd


# In[101]:


df = pd.read_csv("C:/Users/Dipanjan/Desktop/bank-additional-full.csv", sep = ";")
#df.to_csv("C:/Users/Dipanjan/Desktop/bank-additional-full1.csv", sep = ",")


# In[102]:


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


# In[103]:


#encoding independent categorical variables
columns = ['job' ,'poutcome', 'marital' , 'education' , 'default' , 'housing' , 'loan' , 'contact' , 'month' , 'day_of_week']

df_final = category_onehot_multcols(columns , df)
df_final


# In[104]:


#encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_encoded = le.fit_transform(df['y'])


# In[105]:


df_final.drop('y' , axis=1 , inplace=True)


# In[106]:


#splitting into test , train 


#df_Y = Y_encoded
#print(df_Y.dtypes)
from sklearn.model_selection import train_test_split
df_X_train, df_X_test , df_Y_train , df_Y_test = train_test_split(df_final, Y_encoded, test_size = 0.2)


# In[118]:


#model performance
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(df_X_train , df_Y_train)
Y_pred = model.predict(df_X_test )


# In[119]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
cm=confusion_matrix(df_Y_test,Y_pred)
accuracy_score=accuracy_score(df_Y_test,Y_pred)
precision_score=precision_score(df_Y_test,Y_pred)
recall_score=recall_score(df_Y_test,Y_pred)
print (f'Confusion matrix: {cm}')
print (f'Accuracy score: {accuracy_score}')
print (f'Precision score: {precision_score}')
print (f'Recall score: {recall_score}')


# In[ ]:




