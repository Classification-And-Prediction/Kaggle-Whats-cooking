import pandas as pd
import numpy as np
import re, nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer      
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.linear_model import LogisticRegression


train_data_df = pd.read_csv('traindata.csv',delimiter='\t',header = None)
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
corpus_data_features_nd = (corpus_data_features.toarray())
print corpus_data_features_nd.shape

my_model = LinearSVC(penalty = 'l1',dual=False,C=0.9)#,loss='hinge')
#my_model = LogisticRegression(penalty = 'l2', C= 0.2)
#my_model = KNeighborsClassifier()
my_model = my_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Cuisine)
test_pred = my_model.predict(corpus_data_features_nd[len(train_data_df):])

spl = []
for i in range(len(test_pred)) :
    spl.append(i)

fw = open("results.csv","w")
fw.write("id,cuisine"+"\n")
for ids, cus in zip(test_data_df.Id[spl], test_pred[spl]):
	fw.write(str(ids)+","+str(cus)+"\n")
