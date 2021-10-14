#!/usr/bin/env python3
"""reducer.py"""
from operator import itemgetter
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score as As
import warnings 
warnings.filterwarnings("ignore")
y = []
c = 0
container = []
row = []
for line in sys.stdin:
 line = line.strip()
 count,word = line.split('\t', 1)
 words = word.split(",")
 np_words = np.array(words).astype("float")
 container.append(np_words)
df = pd.DataFrame(container)
"""print(df)"""
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
y = y.astype(int)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2 , random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("KNN Model")
print(cr(y_test,neigh.predict(X_test)))
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print("Logistic Regression model")
print(cr(y_test,clf.predict(X_test)))
from xgboost import XGBClassifier
xg = XGBClassifier(eval_metric="logloss")
xg.fit(X_train, y_train)
print("XGB Model")
print(cr(y_test,xg.predict(X_test)))


