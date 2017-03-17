import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_traiin, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_traiin,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)
example_measures = np.array([3,4,4,2,7,3,3,2,8])
prediction = clf.predict(example_measures)
print(prediction)