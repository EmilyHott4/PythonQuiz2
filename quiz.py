import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

with open('animals_train.txt') as f:
    lines = [line.strip() for line in f]
   
feature = lines[0].split(',')
data = [[int(item) for item in items.split(',')] for items in lines[1:]]
animals_train = pd.DataFrame(data, columns = feature)
print('Shape: ',animals_train.shape)
animals_train.head()

with open('animals_test.txt') as f:
    lines = [line.strip() for line in f]
   
feature = lines[0].split(',')[1:]
data = [[int(item) for item in items.split(',')[1:]] for items in lines[1:]]
animal_name = [item.split(',')[:1][0] for item in lines[1:]]
animals_test = pd.DataFrame(data, columns = feature)
animals_test['animal_name'] = animal_name
animals_test.head()


# reading the file
with open('Classes.txt') as f:
    lines = [line.strip() for line in f]
   
features = lines[0].split(',')[1:]
data1 = [item.split(',')[:3] for item in lines[1:]]
data2 = [item.split(',')[3:] for item in lines[1:]]
Classes = pd.DataFrame(data1,columns=features)
Classes['Animal_Names'] = data2
Classes


# reading the file
with open('predictions.txt') as f:
    lines = [line.strip() for line in f]
   
features = lines[0].split(',')
data = [item.split(',') for item in lines[1:]]
predictions = pd.DataFrame(data,columns=features)
predictions.head()


X_train = animals_train.iloc[:,:-1].values
y_train = animals_train.iloc[:,-1:].values
X_test = animals_test.iloc[:,:-1].values

clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

y_pred = clf.predict(X_test)
animals_test['predicted'] = y_pred
animals_test = animals_test[['animal_name','predicted']]
# saving to disk
animals_test.to_csv('animals_test.csv', index=False)
animals_test.head()
