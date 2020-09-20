import pandas as pd , numpy as np , matplotlib.pyplot as plt , seaborn as sns
df=pd.read_csv("C:/Users/Dipanjan/Desktop/Python/Kaggle/House Prices/train.csv")

#missing values
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['Fireplaces']=df['Fireplaces'].fillna(df['Fireplaces'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])
#df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])

#dropping features with large missing data
df.drop(['PoolQC'],axis=1,inplace=True)
df.drop(['MiscFeature'],axis=1,inplace=True)
df.drop(['Alley'],axis=1,inplace=True)
df.drop(['Fence'],axis=1,inplace=True)
df.drop(['FireplaceQu'],axis=1,inplace=True)

df=df.values
#handling categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
#df[:, 2,[5:16],[20:25],[26:33],34,[38:42],52,54,56,58,[61:63],73,74]=labelencoder.fit_transform(df[:,2,[5:16],[20:25],[26:33],34,[38:42],52,54,56,58,[61:63],73,74])
df[:, [2,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,26,27,28,29,30,31,32,34,38,39,40,41,42,52,54,56,58,61,62,73]]=labelencoder.fit_transform(df[:, [2,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,26,27,28,29,30,31,32,34,38,39,40,41,42,52,54,56,58,61,62,73]])
#df[:, 2,5:16,20:25,26:33,34,38:42,52,54,56,58,61:63,73,74]=labelencoder.fit_transform(df[:,2,5:16,20:25,26:33,34,38:42,52,54,56,58,61:63,73,74])
#print(df[:,2].shape)
print (df[:, [2,5,16]].shape)
print (df[:,20:25].shape)
#,26:33,34,38:42,52,54,56,58,61:63,73,74].shape
