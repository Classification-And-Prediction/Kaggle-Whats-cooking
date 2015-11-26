library(tm)
library(data.table)
library(Matrix)
library(caret)
library(SnowballC)
library(xgboost)
library(Ckmeans.1d.dp)

#- load data files and flatten
train_raw = read.csv("traindata.csv", sep="\t", header=FALSE)
test_raw = read.csv("testdata.csv", sep="\t", header=FALSE)

names(train_raw)<-c('id','cuisine','ingredients')
names(test_raw)<-c('id','ingredients')

#- create a matrix of ingredients in both the TRAIN and test set
train_ingredients <- Corpus(VectorSource(train_raw$ingredients))
train_ingredients <- tm_map(train_ingredients, content_transformer(tolower), mc.cores=1)
train_ingredients <- tm_map(train_ingredients, removePunctuation, mc.cores=1)
train_ingredients <- tm_map(train_ingredients, function(x)removeWords(x,stopwords()), mc.cores=1)

test_ingredients <- Corpus(VectorSource(test_raw$ingredients))
test_ingredients <- tm_map(test_ingredients, content_transformer(tolower), mc.cores=1)
test_ingredients <- tm_map(test_ingredients, removePunctuation, mc.cores=1)
test_ingredients <- tm_map(test_ingredients, function(x)removeWords(x,stopwords()), mc.cores=1)

#- alternative: create weighted DTM (e.g. using TF-IDF)
train_ingredientsDTM <- DocumentTermMatrix(train_ingredients, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), stopwords = TRUE, removePunctuation=TRUE)) 
train_ingredientsDTM <- as.data.frame(as.matrix(train_ingredientsDTM))

test_ingredientsDTM <- DocumentTermMatrix(test_ingredients, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), stopwords = TRUE, removePunctuation=TRUE)) 
test_ingredientsDTM <- as.data.frame(as.matrix(test_ingredientsDTM))

train_ingredientsDTM$cuisine <- as.factor(train_raw$cuisine)
																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
xgbmat	<- xgb.DMatrix(Matrix(data.matrix(train_ingredientsDTM[, !colnames(train_ingredientsDTM) %in% c("cuisine")])), label=as.numeric(train_ingredientsDTM$cuisine)-1)

#- train our multiclass classification model using softmax
xgb 	<- xgboost(xgbmat, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																															

#- predict on the test set and change cuisine back to string
xgb.test      <- predict(xgb, newdata = data.matrix(test_ingredientsDTM))
xgb.test.text <- levels(train_ingredientsDTM$cuisine)[xgb.test+1]

#- build and write the submission file
test_match   <- cbind(as.data.frame(test_raw$id), as.data.frame(xgb.test.text))
colnames(test_match) <- c("id", "cuisine")

write.csv(test_match, file = 'result_in_r.csv', row.names=F, quote=F)																								