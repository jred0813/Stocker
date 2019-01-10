# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:43:39 2019

@author: hoopq
"""
import tushare as ts
import pandas as pd 
from sklearn import svm,preprocessing

#獲取銀行資料到電腦
df_CB=ts.get_hist_data('601988', start='2014-01-01', end='2018-01-01')
df_CB=pd.read_csv(r'CB.csv',encoding='gbk')


#以日期作為index來排列
df_CB = df_CB.set_index('date')
df_CB = df_CB.sort_index()

#value表示漲跌

value = pd.Series(df_CB['close']-df_CB['close'].shift(1),\
                  index=df_CB.index)
value = value.bfill()
value[value>=0]=1 
value[value<0]=0 
df_CB['Value']=value

#補空值
df_CB=df_CB.fillna(method='bfill')
df_CB=df_CB.astype('float64')

#選取數據的80％作為訓練集，20％作為測試集
L=len(df_CB)
train=int(L*0.8)
total_predict_data=L-train

#對樣本特徵進行歸一化處理
df_CB_X=df_CB.drop(['Value'],axis=1)
df_CB_X=preprocessing.scale(df_CB_X)

#開始循環預測，每次向前預測一個值
correct = 0
train_original=train
while train<L:
    Data_train=df_CB_X[train-train_original:train]
    value_train = value[train-train_original:train]
    Data_predict=df_CB_X[train:train+1]
    value_real = value[train:train+1]
    #核函數分別選取'ploy','linear','rbf'
    classifier = svm.SVC(C=1.0, kernel='poly') 
    #classifier = svm.SVC(kernel='linear')
    #classifier = svm.SVC(C=1.0,kernel='rbf')
    classifier.fit(Data_train,value_train)
    value_predict=classifier.predict(Data_predict)
    print("value_real=%d value_predict=%d"%(value_real[0],value_predict))

    #計算測試集中的正確率
    if(value_real[0]==int(value_predict)):
        correct=correct+1
    train = train+1
#輸出準確率
correct=correct*100/total_predict_data
print("Correct=%.2f%%"%correct)

Data = df_CB[['open','close']]
Data=Data.astype(float)
Data.plot()
Data.ix['2015-01-01':'2018-12-01'].plot()




