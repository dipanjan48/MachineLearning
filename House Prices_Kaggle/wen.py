import numpy as np
import pandas as pd
import seaborn as sns
import xgboost

df_train = pd.read_csv("C:/Users/Dipanjan/Desktop/Python/Kaggle/House Prices/train.csv")
df_test = pd.read_csv("C:/Users/Dipanjan/Desktop/Python/Kaggle/House Prices/test.csv")
main_df_test = df_test.copy()
main_df_train = df_train.copy()
#main_df_test.to_csv("C:/Users/Dipanjan/Desktop/Python/Kaggle/House Prices/submission.csv", index = False)

shape_train = df_train.shape
shape_test = df_test.shape


#sns.heatmap(df_train.isnull() , yticklabels=False , cmap='viridis')

#Fit missing values
df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].mode())
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna(df_train['FireplaceQu'].mode()[0])
df_train['BsmtQual'] = df_train['BsmtQual'].fillna(df_train['BsmtQual'].mode()[0])
df_train['BsmtCond'] = df_train['BsmtCond'].fillna(df_train['BsmtCond'].mode()[0])
df_train['GarageType'] = df_train['GarageType'].fillna(df_train['GarageType'].mode()[0])
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].mode()[0])
df_train['GarageFinish'] = df_train['GarageFinish'].fillna(df_train['GarageFinish'].mode()[0])
df_train['GarageQual'] = df_train['GarageQual'].fillna(df_train['GarageQual'].mode()[0])
df_train['GarageCond'] = df_train['GarageCond'].fillna(df_train['GarageCond'].mode()[0])
df_train['BsmtCond'] = df_train['BsmtCond'].fillna(df_train['BsmtCond'].mode()[0])
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna(df_train['BsmtFinType1'].mode()[0])
df_train['BsmtFinType2'] = df_train['BsmtFinType2'].fillna(df_train['BsmtFinType2'].mode()[0])
df_train['MasVnrType'] = df_train['MasVnrType'].fillna(df_train['MasVnrType'].mode()[0])
df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mode()[0])
df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])

df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mode())
df_test['FireplaceQu'] = df_test['FireplaceQu'].fillna(df_test['FireplaceQu'].mode()[0])
df_test['BsmtQual'] = df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])
df_test['BsmtCond'] = df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['GarageType'] = df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].mode()[0])
df_test['GarageFinish'] = df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['GarageQual'] = df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])
df_test['GarageCond'] = df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_test['BsmtCond'] = df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])
df_test['MasVnrType'] = df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mode()[0])
df_test['Electrical'] = df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['Utilities'] = df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mode()[0])
df_test['Functional'] = df_test['Functional'].fillna(df_test['Functional'].mode()[0])
df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mode()[0])
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].mode()[0])
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mode()[0])
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mode()[0])
df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['GarageCars'] = df_test['GarageCars'].fillna(df_test['GarageCars'].mode()[0])
df_test['GarageArea'] = df_test['GarageArea'].fillna(df_test['GarageArea'].mode()[0])
df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])

#Dropping values with many(above 50%) missing values
df_train.drop(['PoolQC'], axis=1, inplace=True)
df_test.drop(['PoolQC'], axis=1, inplace=True)

df_train.drop(['MiscFeature'], axis=1, inplace=True)
df_test.drop(['MiscFeature'], axis=1, inplace=True)

df_train.drop(['Alley'], axis=1, inplace=True)
df_test.drop(['Alley'], axis=1, inplace=True)

df_train.drop(['Fence'], axis=1, inplace=True)
df_test.drop(['Fence'], axis=1, inplace=True)

df_train.drop(['Id'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)


j_train = df_train.shape
j_test = df_test.shape

#storing categorical variables
columns = ['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

#function to apply onehot-encoder
def category_onehot_multcols(multcolumns,final_df):
    #df_final=final_df
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

#function to concat training and test data set and then apply one-hot-encoder (function: onehot_multicol)
df_concat = pd.concat([df_train , df_test] ,  axis = 0)
#print (df_concat.shape)

df_concat_onehotencode = category_onehot_multcols(columns , df_concat)
#print (columns)
#print (df_concat_onehotencode.shape)
final_df = df_concat_onehotencode.loc[:,~df_concat_onehotencode.columns.duplicated()]
#print (final_df.shape)
df_train_new = final_df.iloc[:1460,:]
df_test_new = final_df.iloc[1460:,:]

X_train =  df_train_new.drop(['SalePrice'] , axis = 1)
Y_train =  df_train_new['SalePrice']
X_test = df_test_new.drop(['SalePrice'] , axis = 1)
#X_test.to_csv("C:/Users/Dipanjan/Desktop/Python/Kaggle/House Prices/1.csv", index = False)
a = X_test.max()
a1 = X_train.max()

#testing model performance--> splittng X_train , Y_train 
from sklearn.model_selection import train_test_split
X_perf_train , X_perf_test , Y_perf_train , Y_perf_test = train_test_split(X_train , Y_train , test_size = 0.2)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=18)
model.fit(X_train , Y_train)
Y_pred = model.predict(X_test)

#preparing submission file ,format and submit
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('sample_submission.csv')
dataset=pd.concat(sub_df['Id'],pred,axis=1)
dataset.columns=['Id','SalesPrice']
dataset.to_csv('sample_submission.csv',index=False)



