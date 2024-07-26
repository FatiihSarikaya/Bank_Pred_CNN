import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#START





#READING THE DATA
pd.set_option('display.max_columns', None)
data = pd.read_csv('../data/Churn_Modelling.csv')

#Getting info about the data and the columns before we go in
#We need to clean the data before we do things on it
("""
print(data.info())
print(data.describe())
print(data.head(3))
print(data.columns)
print(data.sample(5))
""")

#DATA CLEANING


#CUSTOMER ID - SURNAME - RowNumber  AND GEOGRAPHY WILL DROP AND GENDER  WILL BE BINARY (0-1)
#Target col is Exited

#DROP
data = data.drop(['CustomerId','Surname','Geography','RowNumber'], axis=1)


#MAKING BINARY

data['Gender'] = data['Gender'].replace({'Female': 1, 'Male' : 0})


#WE NEED TO SCALE THE DATA
# CreditScore-Tenure-Balance-NumOfProducts-EstimatedSalary need to Scale

#SCALE

cols_to_scale = ['CreditScore','Tenure','Balance','NumOfProducts','EstimatedSalary']
scaler = MinMaxScaler()



data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
print(data[['CreditScore','Tenure','Balance','NumOfProducts','EstimatedSalary']])


#SETTING THE TARGET AND TRAIN DATA
#TARGET COLUMN
print(data['Exited'].unique())

X = data.drop('Exited', axis=1)
y = data['Exited']


#TRAIN TEST SPLIT THE DATA

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)



#CREATING THE MODEL

model = keras.Sequential([
    keras.layers.Dense(9,input_shape =(9,), activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer = 'adam',
              loss ='binary_crossentropy',
              metrics =['accuracy']
              )


#FIT THE MODEL

model.fitted = model.fit(X_train,y_train,epochs = 150)
print("fitting results",model.fitted)

#EVALUATE THE MODEL

model_evaluated = model.evaluate(X_test, y_test)
print("Model Evaluated ", model_evaluated)

#MODEL PREDICTS

yp = model.predict(X_test)
print("Model predict: ", yp[:10])


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(y_test[:10])
print(y_pred[:10])


#CLASSIFICATION REPORT

report = classification_report(y_test,y_pred,output_dict=True)


#CHECHKING FROM THE HEATMAP FIGURE THAT HOW IS THE PREDICT

cm = tf.math.confusion_matrix(labels = y_test, predictions = y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


#SHOWING THE RESULTS OF THE CLASSIFICATION REPORT

print("Classification REPORT: ", report)

print("Accuracy score of Classification REPORT: \n", report['accuracy'])

print("F1 score of Classification REPORT: \n", report['macro avg']['f1-score'])

print("Recall score of Classification REPORT: \n", report['macro avg']['recall'])



#END





