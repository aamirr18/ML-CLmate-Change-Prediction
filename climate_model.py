#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing required libraries for the model
import pandas as pd

# Feature Scaling libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Machine learning algorithm
from sklearn.ensemble import GradientBoostingRegressor

#For Pickle
import pickle

## Importing Dataset
df = pd.read_csv("C:\\Users\\admin\\yield_df.csv") #Change the filepath

# Categorical value treatment
df_1 = pd.get_dummies(df, columns=['Area',"Item"], prefix = ['Country',"Item"])
# Splitting the data into X & Y
x=df_1.loc[:, df_1.columns != 'hg/ha_yield']
y=df['hg/ha_yield']


#Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x) 

## Fitting the model
gbr = GradientBoostingRegressor(max_depth=4, min_samples_leaf=4, min_samples_split=10, n_estimators=150)
gbr.fit(x, y)

## Saving model to disk
pickle.dump(gbr, open('climate_model.pkl','wb'))

