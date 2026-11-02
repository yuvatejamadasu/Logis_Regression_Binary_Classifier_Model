'''
in this file we explain the logistic regression algorithm using the binary classification data
the datset we used is breast cancer dataset.
'''

#import all the librarys

import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')


#take the dataset

df=pd.read_csv("breast_cancer_dataset.csv")

#remove unwanted columns in the dataset

column_names=df.columns

df=df.drop(['id','Unnamed: 32'],axis=1)

#in the dependent column is a text values convert that into  a numbers using map method

#to find the unique values

dpen_coln_uni=df['diagnosis'].unique()

df['diagnosis']=df['diagnosis'].map({'M':0,'B':1}).astype(int)



#if it has any null values remove the null values in the data

rem_null=df.dropna()

#now  X in a independent columns and y in a dependent columns

X=df.iloc[ : , 1: ]
y=df.iloc[ : , 0]


#now data is ready split the data into a training part and testing part

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#give data to the algorithm(Logistic Regression)

reg=LogisticRegression()
reg.fit(x_train,y_train)

#Training Performance

#to find the predicted values

train_predict=reg.predict(x_train)

#to find the confusion_matrix of the training data
print(f"the training data confusion_matrix was:")
print(f"{confusion_matrix(y_train,train_predict)}")
#to find the accuracy score of the training data
print(f"the training data accuracy score was:")
print(f"{accuracy_score(y_train,train_predict)}")
#to find the classification_report of the training data
print(f"the classification report of the training data was:")
print(f"{classification_report(y_train,train_predict)}")



#Testing Performance

#to find the predicted values

test_predict=reg.predict(x_test)

#to find the confusion_matrix of the training data
print(f"the testing data confusion_matrix was:")
print(f"{confusion_matrix(y_test,test_predict)}")
#to find the accuracy score of the testing data
print(f"the testing data accuracy score was:")
print(f"{accuracy_score(y_test,test_predict)}")
#to find the classification_report of the testing data
print(f"the classification report of the testing data was:")
print(f"{classification_report(y_test,test_predict)}")



#check own data point

print(f'to check the own data prediction')

#in the total indpendent columns is 30 (5 X 6)

values=np.random.random((5,6)).ravel()

own_data_pred=reg.predict([values])[0]
print(own_data_pred)


#check own data prediction

with open("Logistic_Regression_Binary.pkl","wb") as f:
    pickle.dump(reg,f)

#load the model for check the predcition the model is file is worked or not

with open("Logistic_Regression_Binary.pkl","rb") as p:
    m=pickle.load(p)

model_predict=m.predict([values])[0]
print(model_predict)
