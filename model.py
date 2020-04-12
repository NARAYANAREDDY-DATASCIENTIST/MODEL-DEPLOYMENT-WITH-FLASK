# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 09:54:18 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""
# IMPORT LIBRARIES

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# READ THE DATASET
dataset=pd.read_csv('hiring.csv')

dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

x=dataset.iloc[:,:3]
y=dataset.iloc[:,3]

# build the linear regression model
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
# fit the model

regression.fit(x,y)

# saving model to disk
pickle.dump(regression, open('model.pkl','wb'))

# loading model to compare results
model=pickle.load(open('model.pkl','rb'))

               
