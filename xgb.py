import pandas as pd
import numpy as np
import re, nltk
from nltk.stem import WordNetLemmatizer      
from sklearn.feature_extraction.text import *
import xgboost as xgb

train_data_df = pd.read_csv('traindata3.csv',delimiter='\t',header = None)
test_data_df = pd.read_csv('testdata.csv',header = None ,delimiter="\t")

train_data_df.columns = ["Id","Cuisine","Ingredients"]
test_data_df.columns = ["Id","Ingredients"]

train_data_df1 = train_data_df.drop('Id',1)
test_data_df1 = test_data_df.drop('Id',1)

stemmer = WordNetLemmatizer()

def stem_tokens(tokens, stemmer):
	stemmed = []
	for item in tokens:
		stemmed.append(stemmer.lemmatize(item))
	return stemmed

def tokenize(text):
    
	text = re.sub("[^a-zA-Z]", " ", text)
	text = re.sub(" +"," ", text)
	tokens = nltk.word_tokenize(text)
	stems = stem_tokens(tokens, stemmer)
	return stems

vectorizer = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english')

corpus_data_features = vectorizer.fit_transform(train_data_df1.Ingredients.tolist() + test_data_df1.Ingredients.tolist())

labels_numeric = pd.Series(train_data_df1.Cuisine, dtype = "category")

labels_numeric = labels_numeric.astype(np.float)

xg_train = xgb.DMatrix(corpus_data_features[0:len(train_data_df1)], label = labels_numeric)
xg_test = xgb.DMatrix(corpus_data_features[len(train_data_df):])

param = {}
param['objective'] = 'multi:softmax'
#param['objective'] = 'multi:softprob'
param['eta'] = 0.1
param['max_depth'] = 10
param['num_class'] = 20
num_round = 5000

gbm = xgb.train(param,xg_train,num_round)
test_pred = gbm.predict(xg_test)

spl = []
for i in range(len(test_pred)) :
    spl.append(i)

fw = open("resultXGB10.txt","w")
for ids, cus in zip(test_data_df.Id[spl], test_pred[spl]):
	fw.write(str(ids)+","+str(cus)+"\n")


