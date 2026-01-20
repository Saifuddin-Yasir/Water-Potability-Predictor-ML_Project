import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
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
pipeline= Pipeline([('scaler',RobustScaler())])
x_train= pipeline.fit_transform(x_train)
x_test= pipeline.transform(x_test)
best_model = XGBClassifier(learning_rate= 0.1,max_depth=5,n_estimators= 200)
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)

accuracy_score=accuracy_score(y_test, y_pred)
f1_score=f1_score(y_test, y_pred)
precision_score=precision_score(y_test, y_pred)
recall_score=recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy_score}')
print(f'F1 Score: {f1_score}')
print(f'Precision Score: {precision_score}')
print(f'Recall Score: {recall_score}')

with open('Water _Potability_Model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

    