import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


data=pd.read_csv('animals_train.csv')
test_d=pd.read_csv('animals_test.csv')
resutls=pd.read_csv('Classes.csv')
feature_cols = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','class_number']
featurecols = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize',]

x_data = data[feature_cols] 
y_data = data.label
xdata = test_d[featurecols] 
ydata = test_d.label

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.01, random_state=42)
Xtrain, X_test, ytrain, y_test = train_test_split(xdata, ydata, test_size=0.99, random_state=42)

columns=list(x_data.colums)
onehot= OneHotEncoder()
clt = ColumnTransformer([('binarize',onehot,columns)], remainder='passthrough')

X_train=clt.fit_transform(X_train)

X_test=clt.transform(X_test)

## using KNN
test_score=[]
for i in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_hat=knn.predict(X_test)
    score= 1-accuracy_score(y_test,y_hat)
    test_score.append(score)

Nval=list(range(1,40))
param_grid={'n_neighbors':Nval}
model = GridSearchCV(knn,param_grid=param_grid, cv=10, scoring='accuracy')

model.fit(X_train,y_train)
grid_cv_predict = model.predict(X_test)
print(accuracy_score(y_test,pred),'\n')

for i in resutls:
    if grid_cv_predict==i.Class_Number[0]:
        print(i)
