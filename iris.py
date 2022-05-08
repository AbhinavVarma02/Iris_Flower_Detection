import pandas as pd
import numpy as np
import pickle
import csv 

import os
os.getcwd
os.chdir('C:\\Users\\ABHI ALEXY\\Desktop\\vs code\\IRIS_Flower_Detection')

df=pd.read_csv('iris.data.csv')
print(df)

X=np.array(df.iloc[:,0:4])
y=np.array(df.iloc[:,4:])

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train,y_train)

pickle.dump(sv,open('iri.pkl','wb'))