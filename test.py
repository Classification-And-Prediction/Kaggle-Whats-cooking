import pandas as pd
import numpy as np
import re, nltk        
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

train_data_df = pd.read_csv('traindata.csv',delimiter='\t',header = None)
test_data_df = pd.read_csv('testdata.csv',header = None ,delimiter="\t")

train_data_df.columns = ["Id","Cuisine","Ingredients"]
test_data_df.columns = ["Id","Ingredients"]

train_data_df1 = train_data_df.drop('Id',1)
test_data_df1 = test_data_df.drop('Id',1)

vectorizer = TfidfVectorizer( lowercase=False )

corpus_data_features = vectorizer.fit_transform(train_data_df1.Ingredients.tolist() + test_data_df1.Ingredients.tolist())
corpus_data_features_nd = (corpus_data_features.toarray())
#print corpus_data_features_nd.shape

X_train, X_test, y_train, y_test  = train_test_split(corpus_data_features_nd[0:len(train_data_df)],train_data_df.Cuisine, random_state=2)
"""
my_model = LinearSVC(penalty = 'l2')#loss='hinge')
Cs = np.logspace(-6, -1, 10)
#pl = np.array(['l1','l2'])
du = np.array([True])
#my_model = LogisticRegression(penalty = 'l1')
clf = GridSearchCV(estimator = my_model, param_grid = dict(C=Cs, dual = du))
clf.fit(corpus_data_features_nd[0:len(train_data_df)], train_data_df.Cuisine)        
print clf.best_score_                            
print clf.best_estimator_.C
try :
	print clf.best_estimator_.penalty
except :
	pass
print clf.score(corpus_data_features_nd[0:len(train_data_df)], train_data_df.Cuisine)      
"""
for i in range(3,11) :
	a = float(i)/10
	my_model = LogisticRegression(penalty = 'l1',C = a )
	#my_model = my_model.fit(X=X_train, y=y_train)
	#test_pred = my_model.predict(X_test)
	print a
	print cross_val_score(my_model,X_train,y_train,cv=10,scoring='accuracy').mean()
"""
accu_score = cross_val_score(my_model,corpus_data_features_nd[0:len(train_data_df)],train_data_df.Cuisine,cv=10,scoring='accuracy').mean()
print "\n"
print "accuracy score : ",accu_score  
"""

"""
#my_model = LinearSVC(penalty = 'l2',dual = True,C=0.7,loss='hinge')
#my_model = LogisticRegression(penalty = 'l1')
my_model = KNeighborsClassifier()
my_model = my_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Cuisine)
test_pred = my_model.predict(corpus_data_features_nd[len(train_data_df):])

spl = []
for i in range(len(test_pred)) :
    spl.append(i)

fw = open("results3.txt","w")
for ids, cus in zip(test_data_df.Id[spl], test_pred[spl]):
	fw.write(str(ids)+","+str(cus)+"\n")
"""