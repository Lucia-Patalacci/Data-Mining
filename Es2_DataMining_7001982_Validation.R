###############  ESERCIZIO 2  ###############

####### PREDIRE VALORI DEFAULT########

#Import e preparazione data set (training e test set)##

setwd("~/Documents/Curriculum/MABIDA")
library(readr)
es2_train_7001982 <- read.csv("es2_train_7001982.csv", header=T,sep=";", na.string="?")
es2_test <- read.csv("es2_test.csv", header=T,sep=";", na.string="?")

#escludo col ID da training e test set#
es2_train_7001982<-es2_train_7001982 [,setdiff(colnames(es2_train_7001982),"ID")]
es2_test<-es2_test [,setdiff(colnames(es2_test),"ID")]
n <- nrow(es2_train_7001982)

# analisi dataset
table(es2_train_7001982$DEFAULT)
summary(es2_train_7001982)
prop.table(table(es2_train_7001982$DEFAULT))
barplot(table(es2_train_7001982$DEFAULT)/length(es2_train_7001982$DEFAULT),
        xlab = "Default", ylab="Frequenza", main = "Default: Freq.Relativa", col=c("lightgrey","lightblue"))

#SCATTERPLOT PER PRED.NUMERICI#
library(psych)
pairs.panels(es2_train_7001982[c("pay_1","pay_2","pay_3","pay_4","pay_5","pay_6")]
             ,main="Scatterplot Matrix")
pairs.panels(es2_train_7001982[c("bill_amt1","bill_amt2","bill_amt3","bill_amt4","bill_amt5","bill_amt6")]
             ,main="Scatterplot Matrix")
pairs.panels(es2_train_7001982[c("pay_amt1","pay_amt2","pay_amt3","pay_amt4","pay_amt5","pay_amt6")]
             ,main="Scatterplot Matrix")
pairs.panels(es2_train_7001982[c("age","limit_bal","pay_6","bill_amt6","pay_amt6")]
             ,main="Scatterplot Matrix")

####### MULTIPLE LOGISTIC REGRESSION################
#split del training set per il validation set 

n <- nrow(es2_train_7001982)
n_train <- ceiling(n*3/4)
set.seed(123)
index_train <- sample(1:n,n_train,replace=F)     ## indici di riga del training set
index_valid <- setdiff(1:n,index_train)           ## indici di riga del validation set
es2_train <- es2_train_7001982[index_train,]      ## training set
es2_valid <- es2_train_7001982[index_valid,]        ## validation set

library(ISLR)
library(MASS)
library(nnet)
logis_m0 <- glm(DEFAULT~1,family="binomial", data=es2_train)
logis_m1 <- glm(DEFAULT~.,family="binomial", data=es2_train)

#selezione con algoritmo Stepwise Both#####
logis_learn_aic <- stepAIC(logis_m1,scope=list(lower=logis_m0,upper=logis_m1),direction="both",trace=F)
logis_learn_bic <- stepAIC(logis_m1,scope=list(lower=logis_m0,upper=logis_m1),direction="both",k=log(n),trace=F)
logis_learn_trv <- stepAIC(logis_m1,scope=list(lower=logis_m0,upper=logis_m1),direction="both",k=qchisq(0.95,1),trace=F)

summary(logis_learn_aic)
#confint.default(logis_learn_aic)
#exp(confint.default(logis_learn_aic)[-1,])
summary(logis_learn_bic)
#confint.default(logis_learn_bic)
#exp(confint.default(logis_learn_bic)[-1,])
summary(logis_learn_trv)
#confint.default(logis_learn_trv)
#exp(confint.default(logis_learn_trv)[-1,])

# classi (probabilità) predette  P=(Y=1/X)
cl_pred_aic <- predict(logis_learn_aic,es2_test,type="response")
cl_pred_bic <- predict(logis_learn_bic,es2_test,type="response")
cl_pred_trv <- predict(logis_learn_trv,es2_test,type="response")

############### 1.classificazione k Nearest Neighbors  ###############

# nome della variabile di classificazione
classVar <- "DEFAULT"
es2_train_7001982[,classVar] <- factor(es2_train_7001982[,classVar])

# nomi dei predittori (automatico)
numVar <- c()
for(i in 1:ncol(es2_train_7001982)) {
  if(is.numeric(es2_train_7001982[,i])) {
    numVar <- c(numVar,colnames(es2_train_7001982)[i])
  }
}
catVar <- setdiff(colnames(es2_train_7001982),c(numVar,classVar))

#funzione per spit in training e validation set
evsplitFun <- function(x.class,data,p=0.75) {
  n <- nrow(data)
  n_train <- ceiling(n*p)
  index_train <- sample(1:n,n_train,replace=F)
  index_test <- setdiff(1:n,index_train)
  allVar <- setdiff(colnames(data),x.class)
  OUT <- list(train_set=data[index_train,allVar],
              test_set=data[index_test,allVar],
              train_label=data[index_train,x.class],
              test_label=data[index_test,x.class]
  )
  OUT
}
# split dati per validazione esterna
set.seed(123)
evList <- evsplitFun(x.class=classVar,data=es2_train_7001982,p=0.75)

## RISCALATURA DI UN PREDITTORE NUMERICO
#    Argomenti:
#     - 'x': un vettore di valori numerici
#     - 'type': 'z-score' per standardizzazione, 'min-max' per normalizzazione min-max
#    Output: i valori numerici riscalati
rescalFun <- function(x,type="z-score") {
  if(type=="z-score") {
    res <- (x-mean(x,na.rm=T))/sd(x,na.rm=T)
  } else if(type=="min-max") {
    res <- (x-min(x,na.rm=T))/(max(x,na.rm=T)-min(x,na.rm=T))
  } else {
    stop("Unknown type '",type,"'. Valid values are 'z-score' and 'min-max'")  
  }
  res
}

## CODIFICA DUMMY
#    Argomenti:
#     - 'x.cat': un vettore contenente i nomi dei predittori categorici
#     - 'data': il dataset contenente tutti i predittori (anche quelli numerici)
#    Output: il dataset fornito dopo codifica dummy dei predittori categorici
dummyFun <- function(x.cat,data) {
  code_str <- paste("~",paste(x.cat,collapse="+"),sep="")
  dumdat <- model.matrix(formula(code_str),data=data)
  cbind(data[,setdiff(colnames(data),x.cat),drop=F],dumdat[,-1,drop=F])
}

## PRE_PROCESSING PER K-NN: utilizza rescalFun() e dummyFun()
#    Argomenti:
#     - 'x.num': un vettore contenente i nomi dei predittori numerici
#     - 'x.cat': un vettore contenente i nomi dei predittori categorici
#     - 'rescaling': 'z-score' per standardizzazione, 'min-max' per normalizzazione min-max
#     - 'data': il dataset contenente tutti i predittori
#    Output: il dataset fornito dopo codifica dummy dei pred. categorici e riscalatura dei pred. numerici
preprocFun <- function(x.num=NULL,x.cat=NULL,rescaling="z-score",data) {
  res <- data
  if(length(x.num)>0) res[,x.num] <- apply(res[,x.num,drop=F],2,rescalFun,type=rescaling)
  if(length(x.cat)>0) res <- dummyFun(x.cat=x.cat,data=res)
  res
}
# pre-processing
knn_train <- preprocFun(x.num=numVar,x.cat=catVar,rescaling="z-score",data=evList$train_set)
knn_test <- preprocFun(x.num=numVar,x.cat=catVar,rescaling="z-score",data=evList$test_set)

# k-NN
library(class)
k_thumb <- 2*floor(sqrt(nrow(knn_train))/2)+1

#### funzioni utilizzate per la selezione del modello KNN####
# accuratezza per valori diversi di k
knn_accurFun <- function(train,test,cl_train,cl_test,k,...) {
  res <- vector(length=length(k))
  names(res) <- k
  for(i in 1:length(k)) {
    ipred <- knn(train=train,test=test,cl=cl_train,k=k[i],...)
    res[i] <- cmatFun(cl_test,ipred)$accuracy
  }
  res
}

# accuratezza
cmatFun <- function(actual,predicted) {
  res <- list()
  res[[1]] <- table(actual,predicted,dnn=c("Actual","Predicted"))
  res[[2]] <- round(prop.table(res[[1]])*100,2)
  res[[3]] <- sum(diag(res[[2]]))
  names(res) <- c("absolute","percentage","accuracy")
  res
}
k_seq <- seq(1,100,by=1)
aVet <- knn_accurFun(train=knn_train,test=knn_test,cl_train=evList$train_label,cl_test=evList$test_label,k=k_seq)
a_best <- aVet[which(aVet==max(aVet))]
k_best <- as.numeric(names(a_best))
a_thumb <- aVet[as.character(k_thumb)]
#
plot(aVet~k_seq,type="h",ylab="Accuracy (%)",xlab="Value of k",las=1,cex.lab=1.2,cex.axis=1.1)
abline(h=max(aVet),col="blue",lty=2)
segments(k_best,0,k_best,a_best,col="blue")
abline(h=a_thumb,col="red",lty=2)
segments(k_thumb,0,k_thumb,a_thumb,col="red")

#########selezione modello KNN con KBEST=15
knn_pred <- knn(train=knn_train,test=knn_test,cl=evList$train_label,k=k_best)
knn_cmat <- cmatFun(evList$test_label,knn_pred)
knn_cmat$accuracy
############### Classificazione Naive Bayes  ###############

nb_train <- evList$train_set
nb_test <- evList$test_set

# apprendimento
library(e1071)
nb_learn <- naiveBayes(x=nb_train,y=evList$train_label,laplace=0)
#nb_learn$tables
nb_learn$apriori

# previsione
nb_pred <- predict(nb_learn,es2_test)

# accuratezza
nb_cmat <- cmatFun(evList$test_label,nb_pred)
nb_cmat$accuracy

###############  alberi classificatori  ###############

library(rpart)
tree_learn <- rpart(evList$train_label~.,method="class",data=evList$train_set,
                    parms=list(split="information"),  ## entropia come misura di purity
                    minsplit=20,                      ## numero minimo di esempi per nodo
                    maxdepth=30,                      ## numero massimo di split
                    cp=0.01                           ## parametro di complessita'
)

tree_learn

# previsione
tree_pred <- predict(tree_learn,es2_test,type="class")

# accuratezza
tree_cmat <- cmatFun(evList$test_label,tree_pred)
tree_cmat$accuracy

# grafici CART
library(rpart.plot)
rpart.plot(tree_learn,extra=101,digits = 4,fallen.leaves=TRUE,type=3,roundint=FALSE)

# accuratezza
tree_cmat <- cmatFun(evList$test_label,tree_pred)
tree_cmat$accuracy

#SUMMARY ACCURATEZZA SUI 4 MODELLI DI CLASSIFICAZIONE ##
#accuratezza logistic reg model#
# classi predette con soglia 0.5
#cl_pred<-rep("no",1000)
#cl_pred[cl_pred_aic>0.5]="yes"
cmat_alogis<-table(cl_pred,es2_valid$DEFAULT,dnn=c("Actual","Predicted"))
prop.table(cmat_alogis)

cl_pred<-rep("no",1000)
cl_pred[cl_pred_bic>0.5]="yes"
cmat_blogis<-table(cl_pred,es2_valid$DEFAULT,dnn=c("Actual","Predicted"))
prop.table(cmat_blogis)
tree_alpha <- sum(diag(prop.table(cmat_blogis)))
tree_alpha

cl_pred<-rep("no",1000)
cl_pred[cl_pred_trv>0.5]="yes"
cmat_tlogis<-table(cl_pred,es2_valid$DEFAULT,dnn=c("Actual","Predicted"))
prop.table(cmat_tlogis)

#summary accuratezza su prediction per i 3 modelli##
library(gmodels)
CrossTable(x=evList$test_label,y=knn_pred,prop.chisq=FALSE,prop.c =FALSE,prop.r=FALSE,
           dnn= c("Actual DEFAULT","Predicted DEFAULT"))
CrossTable(x=evList$test_label,y=nb_pred,prop.chisq=FALSE,prop.c =FALSE,prop.r=FALSE,
           dnn= c("Actual DEFAULT","Predicted DEFAULT"))
CrossTable(x=evList$test_label,y=tree_pred,prop.chisq=FALSE,prop.c =FALSE,prop.r=FALSE,
           dnn= c("Actual DEFAULT","Predicted DEFAULT"))

#### Aggiungo CNT predicted a Test-Set###
es2_test_DEFAULT_knn_pred <-cbind(es2_test,knn_pred)
head(es2_test_DEFAULT_knn_pred)

# salvo previsioni in file RData#
save(es2_test_DEFAULT_knn_pred, file = "es2_test_7001982_predicted.Rdata")

