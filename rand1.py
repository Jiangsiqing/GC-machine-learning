# import numpy as np
# import torch.nn as nn
# import torch
# import matplotlib.pyplot as plt
# import pandas as pd
# import WTdelineator as wav
# import wfdb
# import pywt
# dbase = 'mitdb/1.0.0/'
# rec='111'
#
# # rec='100'
# sNum = 0
# sNum2= 1
#
# # When in Windows
# s, att = wfdb.rdsamp(rec,pn_dir=dbase)
# annot = wfdb.rdann(rec, 'atr', pn_dir=dbase)
# sName = att['sig_name']
#
#
# beg1=int(np.floor(0))
# end1= int(np.floor(10000))
#
# beg2=int(np.floor(0))
# end2=int(np.floor(10000))
#
# fs=att['fs']
# sig=s[:,sNum]
# N = sig.shape[0]
# t = np.arange(0,N/fs,1/fs)
# sig=s[0:1000,0]
# coeff=
# plt.subplot(2, 1, 1)
# plt.plot(index[mintime:maxtime], data[mintime:maxtime])
# plt.figure(figsize=(10,6))
# plt.plot(range(1000),sig)
# plt.show()

# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# df=pd.read_csv(r'C:\Users\蒋思清\Desktop\d pd sta\data\train_trts0.7.csv').values
# df2=pd.read_csv(r'C:\Users\蒋思清\Desktop\d pd sta\data\test_trts0.7.csv').values

df=pd.read_csv(r'/data/sjd/d/p_d/stomach/data/train_trts0.7.csv').values
df2=pd.read_csv(r'/data/sjd/d/p_d/stomach/data/test_trts0.7.csv').values
x_train=np.array(df[:,:81])
y_train=np.array(df[:,-1])
model=RandomForestClassifier()
param={'n_estimators':range(1,1000,4),'max_features':range(1,100,2),'max_depth':range(2,15,1),'learning_rate':np.linspace(0.01,2,20),
       'subsample':np.linspace(0.7,0.9,10),'colsample_bytree':np.linspace(0.5,0.98,10),'min_child_weight':range(1,9,1)}
grid_search=GridSearchCV(estimator=model,param_grid=param,cv=5)
x_test=np.array(df2[:,:81])
y_test=np.array(df2[:,-1])
grid_search.fit(x_train,y_train)
# print(model.predict(x_test).shape)
# print(model.predict_proba(x_test))
from sklearn.metrics import recall_score
a=recall_score(y_test,grid_search.predict(x_test),average='micro')
b=accuracy_score(y_test,grid_search.predict(x_test))
cm=confusion_matrix(y_test,grid_search.predict(x_test))
print(a,b)
print(cm)
print(grid_search.best_params_)
print(grid_search.best_estimator_)




