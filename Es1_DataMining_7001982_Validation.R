###############  ESERCIZIO 1  ###############
#######PREDIRE VALORI CNT ########

#Importo e preparo dataset (training e test )
setwd("~/Documents/Curriculum/MABIDA")
library(readr)
es1_train_7001982 <- read.csv("es1_train_7001982.csv", header=T,sep=";", na.string="?")
es1_test <- read.csv("es1_test.csv", header=T,sep=";", na.string="?")
summary(es1_train_7001982)
#escludo col ID da training e test set
es1_train_7001982<-es1_train_7001982 [,setdiff(colnames(es1_train_7001982),"ID")]
es1_test<-es1_test [,setdiff(colnames(es1_test),"ID")]

#split del training set per il validation set 
n <- nrow(es1_train_7001982)
set.seed(1)
n_train <- ceiling(n*3/4)
set.seed(1)
index_train <- sample(1:n,n_train,replace=F) 
index_valid <- setdiff(1:n,index_train)       
es1_train <- es1_train_7001982[index_train,] 
es1_valid <- es1_train_7001982[index_valid,]  

#CHECK PER NORMALITA' CNT
summary(es1_train_7001982$CNT)
hist(es1_train_7001982$CNT)

#CHECK PER CORRELAZIONE VARIABILI NUMERICHE
cor(es1_train_7001982[c("month","weekday","hour","temp","atemp",
                        "hum","windspeed","registered","CNT")])

# SCATTERPLOT matrix
library(psych)
pairs.panels(es1_train_7001982[c("month","weekday","hour","temp","atemp",
                                 "hum","windspeed","registered","CNT")]
             ,main="Scatterplot Matrix")

#REGRESSIONE LINEARE MULTIPLA per alcune analisi#
library(MASS)
library(ISLR)

# modello senza correlata -atemp
summary(m_lin <- lm(CNT~.-atemp,data=es1_train_7001982))

#check leverage statistics##
library(car)
layout(1)
influencePlot(m_lin, id.n=3,id.col="red")

#Variance Inflation Factor per verifica Multicollinearità#
vif(m_lin)

#verifica residui su modello lineare completo#
par(mfrow=c(2,2))
plot(m_lin)

#REGRESSIONE LINEARE MULTIPLA STEPWISE#
library(MASS)
library(ISLR)

# modelli con B=0 e completi
#regressione lineare
m0_lin <- lm(CNT~1,data=es1_train)
m1_lin <- lm(CNT~.,data=es1_train)

#regressione con var.risposta trasformata log(y)
m0_log <- lm(log(CNT)~1,data=es1_train)
m1_log <- lm(log(CNT)~.,data=es1_train)

#regressione poisson
m0_poi <- glm(CNT~1, family="poisson", data=es1_train)
m1_poi <- glm(CNT~., family="poisson", data=es1_train)

#regressione binomiale negativa
m0_nb <- glm.nb(CNT~1,data=es1_train)
m1_nb <- glm.nb(CNT~., data=es1_train)

# selezione stepwise AIC
m_lin_aic <- stepAIC(m1_lin,scope=list(lower=m0_lin,upper=m1_lin),direction="both",trace=F)
m_log_aic <- stepAIC(m1_log,scope=list(lower=m0_log,upper=m1_log),direction="both",trace=F)
m_poi_aic <- stepAIC(m1_poi,scope=list(lower=m0_poi,upper=m1_poi),direction="both",trace=F)
m_nb_aic <- stepAIC(m1_nb,scope=list(lower=m0_nb,upper=m1_nb),direction="both",trace=F)
# selezione stepwise BIC
m_lin_bic <- stepAIC(m1_lin,scope=list(lower=m0_lin,upper=m1_lin),direction="both",k=log(n),trace=F)
m_log_bic <- stepAIC(m1_log,scope=list(lower=m0_log,upper=m1_log),direction="both",k=log(n),trace=F)
m_poi_bic <- stepAIC(m1_poi,scope=list(lower=m0_poi,upper=m1_poi),direction="both",k=log(n),trace=F)
m_nb_bic <- stepAIC(m1_nb,scope=list(lower=m0_nb,upper=m1_nb),direction="both",k=log(n),trace=F)
# selezione stepwise con TRV 
k_trv <- qchisq(0.95,1)
m_lin_trv <- stepAIC(m1_lin,scope=list(lower=m0_lin,upper=m1_lin),direction="both",k=k_trv,trace=F)
m_log_trv <- stepAIC(m1_log,scope=list(lower=m0_log,upper=m1_log),direction="both",k=k_trv,trace=F)
m_poi_trv <- stepAIC(m1_poi,scope=list(lower=m0_poi,upper=m1_poi),direction="both",k=k_trv,trace=F)
m_nb_trv <- stepAIC(m1_nb,scope=list(lower=m0_nb,upper=m1_nb),direction="both",k=k_trv,trace=F)

# sommario selezione
aic.tab <- AIC(m_lin_aic,m_log_aic,m_poi_aic,m_nb_aic)
aic.tab[2,2] <- aic.tab[2,2]+2*sum(log(es1_train_7001982[,"CNT"])) #riporto su stessa scala AIC
aic.tab
bic.tab <- BIC(m_lin_aic,m_log_bic,m_poi_bic,m_nb_bic)
bic.tab[2,2] <- bic.tab[2,2]+2*sum(log(es1_train_7001982[,"CNT"])) #riporto su stessa scala BIC
bic.tab
trv.tab <- AIC(m_lin_trv,m_log_trv,m_poi_trv,m_nb_trv, k=k_trv)
trv.tab[2,2] <- trv.tab[2,2]+2*sum(log(es1_train_7001982[,"CNT"])) #riporto su stessa scala TRV
trv.tab
sel.tab<-cbind(aic.tab,bic.tab,trv.tab)
sel.tab
# previsioni
pred_lin_aic <- predict(m_lin_aic,es1_test)
pred_log_aic <- exp(predict(m_log_aic,es1_test)+0.5*summary(m_log_aic)$sigma^2) # Trasformazione per riportare ai valori originari
pred_poi_aic <- predict(m_poi_aic,es1_test,type="response")
pred_nb_aic <- predict(m_nb_aic,es1_test,type="response")

pred_lin_bic <- predict(m_lin_bic,es1_test)
pred_log_bic <- exp(predict(m_log_bic,es1_test)+0.5*summary(m_log_bic)$sigma^2)
pred_poi_bic <- predict(m_poi_bic,es1_test,type="response")
pred_nb_bic <- predict(m_nb_bic,es1_test,type="response")

pred_lin_trv <- predict(m_lin_trv,es1_test)
pred_log_trv <- exp(predict(m_log_trv,es1_test)+0.5*summary(m_log_trv)$sigma^2)
pred_poi_trv <- predict(m_poi_trv,es1_test,type="response")
pred_nb_trv <- predict(m_nb_trv,es1_test,type="response")

#####  fitting regression trees CART Modeling #####

library(rpart)
# apprendimento
m_tree_01 <- rpart(CNT~.,method="anova",cp=0.001,data=es1_train)
print(m_tree_01)
round(printcp(m_tree_01),5)
ind01 <- which.min(m_tree_01$cptable[, "xerror"])
fit.prune01 <- prune(tree = m_tree_01, cp = m_tree_01$cptable[ind01, "CP"])

# grafici CART
library(rpart.plot)
rpart.plot(fit.prune01,extra=101,digits = 4,fallen.leaves=TRUE,type=3)

# previsione
pred_tree_01 <- predict(fit.prune01,es1_test,type="vector")

#confronto come le previsioni si comportano###
summary(pred_tree_01)
summary(es1_train$CNT)
a<-mean(es1_train$CNT)

cor(es1_valid[,"CNT"],pred_tree_01)

####################  regressione k-nn  ###################
###definisco categorie predittori###
train_numVar <- colnames(es1_train)[sapply(es1_train,is.numeric)]
train_catVar <- setdiff(colnames(es1_train),train_numVar)
test_numVar <- colnames(es1_test)[sapply(es1_test,is.numeric)]
test_catVar <- setdiff(colnames(es1_test),test_numVar)
valid_numVar <- colnames(es1_valid)[sapply(es1_valid,is.numeric)]
valid_catVar <- setdiff(colnames(es1_valid),valid_numVar)

# codifica dummy dei predittori categorici
library(dummies)
es1_knn_train <- dummy.data.frame(es1_train,names=train_catVar,sep="_",drop=F)
es1_knn_test <- dummy.data.frame(es1_test,names=test_catVar,sep="_",drop=F)
es1_knn_valid <- dummy.data.frame(es1_valid,names=valid_catVar,sep="_",drop=F)

# standardizzazione dei predittori numerici z-score
es1_knn_train[,train_numVar]<- scale(es1_knn_train[,train_numVar])
es1_knn_test[,test_numVar]<- scale(es1_knn_test[,test_numVar])
es1_knn_valid[,valid_numVar] <- scale(es1_knn_valid[,valid_numVar])
xnames <- setdiff(colnames(es1_knn_train),"CNT")

#definisco k="ruleofthumb#
k_val <- 2*floor(sqrt(n_train)/2)+1

library(FNN)
knn_learn <- knn.reg(train=es1_knn_train[,xnames],test=es1_knn_test,y=es1_knn_train[,"CNT"],k=k_val)
knn_pred <- knn_learn$pred

cor(es1_valid[,"CNT"],knn_pred$pred)

# grafico valori veri vs predetti
plot(es1_valid[,"CNT"],knn_pred,xlab="Actual",ylab="Predicted")
abline(0,1)

#verifico accuratezza su prediction con i diversi approcci
rmse_calc <- function(obs,exp){ sqrt(mean((obs-exp)^2)) }
mae_calc <- function(obs,exp){ mean(abs(obs-exp)) }
rmse_aic<-c(
  lin_aic=rmse_calc(es1_valid[,"CNT"],pred_lin_aic),
  log_aic=rmse_calc(es1_valid[,"CNT"],pred_log_aic),
  poi_aic=rmse_calc(es1_valid[,"CNT"],pred_poi_aic),
  nb_aic=rmse_calc(es1_valid[,"CNT"],pred_nb_aic)
)
rmse_bic<-c(
  lin_bic=rmse_calc(es1_valid[,"CNT"],pred_lin_bic),
  log_bic=rmse_calc(es1_valid[,"CNT"],pred_log_bic),
  poi_bic=rmse_calc(es1_valid[,"CNT"],pred_poi_bic),
  nb_bic=rmse_calc(es1_valid[,"CNT"],pred_nb_bic)
)
rmse_tree<-c(
  tree_01=rmse_calc(es1_valid[,"CNT"],pred_tree_01),
  tree_knn=rmse_calc(es1_valid[,"CNT"],knn_pred)
)
mae_aic <-c(
  lin_aic=mae_calc(es1_valid[,"CNT"],pred_lin_aic),
  log_aic=mae_calc(es1_valid[,"CNT"],pred_log_aic),
  poi_aic=mae_calc(es1_valid[,"CNT"],pred_poi_aic),
  nb_aic=mae_calc(es1_valid[,"CNT"],pred_nb_aic)
)
mae_bic<-c(
 lin_bic=mae_calc(es1_valid[,"CNT"],pred_lin_bic),
  log_bic=mae_calc(es1_valid[,"CNT"],pred_log_bic),
  poi_bic=mae_calc(es1_valid[,"CNT"],pred_poi_bic),
  nb_bic=mae_calc(es1_valid[,"CNT"],pred_nb_bic)
)
mae_tree<-c(
  tree_01=mae_calc(es1_valid[,"CNT"],pred_tree_01),
  tree_knn=mae_calc(es1_valid[,"CNT"],knn_pred)
)

# accuratezza su prediction con TRV
rmse_trv <- c(
  rmse_calc(es1_valid[,"CNT"],pred_lin_trv),
  rmse_calc(es1_valid[,"CNT"],pred_log_trv),
  rmse_calc(es1_valid[,"CNT"],pred_poi_trv),
  rmse_calc(es1_valid[,"CNT"],pred_nb_trv)
)
mae_trv <- c(
  mae_calc(es1_valid[,"CNT"],pred_lin_trv),
  mae_calc(es1_valid[,"CNT"],pred_log_trv),
  mae_calc(es1_valid[,"CNT"],pred_poi_trv),
  mae_calc(es1_valid[,"CNT"],pred_nb_trv)
)

#####  risultati in tabella  #####

tab.aic<-cbind(aic.tab,rmse_aic,mae_aic)
tab.bic<-cbind(bic.tab,rmse_bic,mae_bic)
tab.trv<-cbind(trv.tab,rmse_trv,mae_trv)
tab.tree<-cbind(rmse_tree,mae_tree)
tab.aic
tab.bic
tab.trv
tab.tree

#### Aggiungo CNT predicted a Test-Set###
es1_test_CNT_lin_bic<-cbind(es1_test,pred_lin_bic)
hist(es1_test_CNT_lin_bic$pred_lin_bic)
head(es1_test_CNT_lin_bic)

# salvo previsioni in file RData#
save(es1_test_CNT_lin_bic, file = "es1_test_7001982_predicted.Rdata")

