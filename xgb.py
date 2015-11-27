import pandas as pd
import numpy as np
import re, nltk
from nltk.stem import WordNetLemmatizer      
from sklearn.feature_extraction.text import *
import xgboost as xgb

train_data_df = pd.read_csv('traindata_XGB.csv',delimiter='\t',header = None)
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

fw = open("resultXGB.csv","w")
fw.write("id,cuisine"+"\n")
for ids, cus in zip(test_data_df.Id[spl], test_pred[spl]):
	cus = str(cus).replace('10.0','indian').replace('11.0','jamaican').replace('12.0','french').replace('13.0','spanish').replace('14.0','russian').replace('15.0','cajun_creole').replace('16.0','thai').replace('17.0','southern_us').replace('18.0','korean').replace('19.0','italian').replace('0.0','irish').replace('1.0','mexican').replace('2.0','chinese').replace('3.0','filipino').replace('4.0','vietnamese').replace('5.0','moroccan').replace('6.0','brazilian').replace('7.0','japanese').replace('8.0','british').replace('9.0','greek')
	fw.write(str(ids)+","+cus+"\n")
