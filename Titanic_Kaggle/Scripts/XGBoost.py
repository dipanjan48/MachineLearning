#XGBoost
import numpy as np, pandas as pd 

#importing dataset
df_train=pd.read_csv("C:/Users/Dipanjan/Desktop/Python/Kaggle/Titanic/train.csv")
#df_test=pd.read_csv("C:/Users/Dipanjan/Desktop/test.csv")
X=df_train.iloc[:,[4,5]].values  #independent variable
Y=df_train.iloc[:,1].values.reshape(-1,1)  #dependent variable

#handling missing values
from sklearn.preprocessing import Imputer
imputer1=Imputer(missing_values='NaN',strategy='median',axis=0)
imputer1=imputer1.fit(X[:,1:2])
X[:,1:2]=imputer1.transform(X[:,1:2])
'''imputer2=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer2=imputer1.fit(X[:,2:3])
X[:,2:3]=imputer1.transform(X[:,2:3])'''

#handling categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,0]=labelencoder.fit_transform(X[:,0])

#split data set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Creating classifier
from xgboost import XGBClassifier
classifier=XGBClassifier(max_depth=5, learning_rate=0.01)  
classifier.fit(X_train,Y_train)

#predicting test results
Y_pred=classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
cm=confusion_matrix(Y_test,Y_pred)
accuracy_score=accuracy_score(Y_test,Y_pred)
precision_score=precision_score(Y_test,Y_pred)
recall_score=recall_score(Y_test,Y_pred)
print (f'Confusion matrix: {cm}')
print (f'Accuracy score: {accuracy_score}')
print (f'Precision score: {precision_score}')
print (f'Recall score: {recall_score}')