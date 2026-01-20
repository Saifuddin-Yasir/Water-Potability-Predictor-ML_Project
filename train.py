import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

data_un= pd.read_csv('water_potability.csv')


imputer_mean= SimpleImputer(strategy='mean')
data_un['ph']= imputer_mean.fit_transform(data_un[['ph']])
data_un['Sulfate']= imputer_mean.fit_transform(data_un[['Sulfate']])
data_un['Trihalomethanes']= imputer_mean.fit_transform(data_un[['Trihalomethanes']])

x= data_un.drop('Potability', axis=1)
y= data_un['Potability']

datas=pd.DataFrame()

for i in data_un.columns:
  q1= data_un[i].quantile(0.25)
  q3= data_un[i].quantile(0.75)
  iqr= q3-q1

  upper= q3+1.5*iqr
  lower= q1-1.5*iqr
  datas[i] = ((data_un[i] >= lower) & (data_un[i] <= upper))

data=datas

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y,random_state=42)

best_model = SVC(
        kernel='rbf',
        C=10,
        gamma=0.1,
        class_weight='balanced',
        random_state=42
    )


pipeline= Pipeline([
   ('Model', best_model)                
                    ])

scaler= RobustScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

accuracyscore=accuracy_score(y_test, y_pred)
f1score=f1_score(y_test, y_pred)
precisionscore=precision_score(y_test, y_pred)
recallscore=recall_score(y_test, y_pred)

print(f'Accuracy: {accuracyscore}')
print(f'F1 Score: {f1score}')
print(f'Precision Score: {precisionscore}')
print(f'Recall Score: {recallscore}')

with open('Water _Potability_Model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

    