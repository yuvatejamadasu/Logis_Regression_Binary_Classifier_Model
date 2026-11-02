'''
in this file is development file we write the code in a oops code in Logistic Regression algorithm
 using the breast cancer dataset
'''

#import all the librarys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')


#creating the class

class LOGISTIC:
    def __init__(self,data):
        #take the dataset
        self.df=pd.read_csv(data)

        self.df = self.df.drop(['id', 'Unnamed: 32'], axis=1)

        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 0, 'B': 1}).astype(int)

        # X in a independent column and y in dependent column
        self.X=self.df.iloc[ : , 1 : ]
        self.y=self.df.iloc[ : , 0]

        # now we split the data into training part and testing part

        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.3,random_state=42)

    #Train Performance
    def train(self):
        #give data to algo(navie_bayers)
         try:
             self.reg=LogisticRegression()
             self.reg.fit(self.x_train,self.y_train)

             #to predict the values
             self.train_predict=self.reg.predict(self.x_train)

             print(f"the confusion_matrix in a training data was:")
             print(f"{confusion_matrix(self.y_train,self.train_predict)}")

             print(f"the accuracy score in a training data was:")
             print(f"{accuracy_score(self.y_train, self.train_predict)}")

             print(f"the confusion_matrix in a training data was:")
             print(f"{classification_report(self.y_train, self.train_predict)}")



         except Exception as e:

             er_ty, er_msg, er_line = sys.exc_info()

             print(f"the error type is :{er_ty} and the error msg is :{er_msg} the error line is :{er_line}")

    # Test Performance
    def test(self):
        # give data to algo(navie_bayers)
        try:

            # to predict the values
            self.test_predict = self.reg.predict(self.x_test)

            print(f"the confusion_matrix in a testing data was:")
            print(f"{confusion_matrix(self.y_test, self.test_predict)}")

            print(f"the accuracy score in a testing data was:")
            print(f"{accuracy_score(self.y_test, self.test_predict)}")

            print(f"the confusion_matrix in a testing data was:")
            print(f"{classification_report(self.y_test, self.test_predict)}")



        except Exception as e:

            er_ty, er_msg, er_line = sys.exc_info()

            print(f"the error type is :{er_ty} and the error msg is :{er_msg} the error line is :{er_line}")


    #save the model

    def save(self):
        try:
            with open("Logistic_Regression_Binary_Breast_Cancer.pkl","wb") as f:
                pickle.dump(self.reg,f)

            with open("Logistic_Regression_Binary_Breast_Cancer.pkl","rb") as p:
                self.m=pickle.load(p)


        except Exception as e:
            er_ty, er_msg, er_line=sys.exc_info()
            print(f"the error eype is :{er_ty} the error msg is :{er_msg} the error line was :{er_line}")


    def check_own_data(self):
        try:
            self.values=np.random.random((5,6)).ravel()

            self.result=self.m.predict([self.values])[0]
            print(f"own data point prediction is :{self.result}")


        except Exception as e:
            er_ty,er_msg,er_line=sys.exc_info()
            print(f"the error type is :{er_ty} the error msg is :{er_msg} the error line is :{er_line}")


if __name__=="__main__":
    dataset_breast_cancer="breast_cancer_dataset.csv"
    obj=LOGISTIC(dataset_breast_cancer)
    obj.train()
    obj.test()
    obj.save()
    obj.check_own_data()

