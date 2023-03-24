##### Coxnet analysis #########
## Katie Merriman 3/7/23 ##

install.packages('survival')
install.packages('survminer')
install.packages('ROCR')
install.packages('pROC')
install.packages('glmnet')
install.packages('precrec')
install.packages('dplyr')
install.packages('Coxnet')
install.packages('sqldf')
install.packages('timeROC')



library(survival)
library(survminer)
library(ROCR)
library(pROC)
library(glmnet)
library(precrec)
library(ROCR)
library(precrec)
library(survival)
library(dplyr)
library(readr)
library(Coxnet)
#library(sqldf)
library(timeROC)

#######################################################
############ LOAD AND PRE-PROCESS DATA ################
#######################################################
f<-"T:\\MIP\\Katie_Merriman\\RadiomicsProject\\Rad_TrainSet.csv"
TrainData <- read.csv(f, header=TRUE, stringsAsFactors=FALSE)

g<-"T:\\MIP\\Katie_Merriman\\RadiomicsProject\\Rad_TestSet.csv"
TestData <- read.csv(g, header=TRUE, stringsAsFactors=FALSE)


h<- "C:\\Users\\merrimankm\\Documents\\RadTest.csv"
data1 <- read.csv(h, header=TRUE, stringsAsFactors=FALSE) 


#checks for missing data
  any(is.na(data1))
  any(is.na(TrainData))
  any(is.na(TestData))

#if any data is missing, find where
  which(is.na(TrainData), arr.ind=TRUE)
  which(is.character(cleanedData), arr.ind=TRUE)

  

#### PRE-PROCESS CONTINUOUS DATA #####  

# Change % positive cores from character to numeric #
  data1$Total_perc <- parse_number(data1$Total_perc)/100
  
  TrainData$Total_perc <- parse_number(TrainData$Total_perc)/100
  TestData$Total_perc <- parse_number(TestData$Total_perc)/100
  
  
# scale all continuous data
  TrainData[c(8:7838,7840:8716)]<-scale(TrainData[c(8:7838,7840:8716)])
  TestData[c(8:7838,7840:8716)]<-scale(TestData[c(8:7838,7840:8716)])
  
  
  
########################################################################
################  RUN COXNET ##########################################
##############################################################
  
 
  dt = sort(sample(nrow(data1), nrow(data1)*.8))
  TrainData<-data1[dt,]
  TestData<-data1[-dt,]
  
  
  trainBCR = sum(TrainData$status == 1, na.rm=TRUE)
  trainLength = nrow(TrainData)-trainBCR
  testBCR = sum(TestData$status == 1, na.rm=TRUE)
  testLength = nrow(TestData)-testBCR
  #trainBCR = sum(TrainData$path_EPE == 1, na.rm=TRUE)
  #trainLength = nrow(TrainData)-trainBCR
  #testBCR = sum(TestData$path_EPE == 1, na.rm=TRUE)
  #testLength = nrow(TestData)-testBCR
 
  statusCheck<-matrix(c(trainBCR,trainLength,testBCR,testLength),ncol=2,byrow=T)
  rownames(statusCheck)<-c("train","test")
  colnames(statusCheck)<-c("BCR","no BCR")
  prop.test(statusCheck)
  
  
#  library(caret)
#  set.seed(3456)
#  trainIndex <- createDataPartition(data1$status, p = .8,
#                                    list = FALSE,
#                                    times = 1)
#  TrainData <- data1[ trainIndex,]
#  TestData <- data1[-trainIndex,]
  
  y = as.matrix(TrainData[c(2:3)])
  
  #y = as.matrix(TrainData[c(2:8717)])
  
  noRad = TrainData[c(8:21)]
  testNoRad = TestData[c(8:21)]
 
 
  #Rad = cbind(TrainData[c(8:21)],TrainData %>% select(matches("T2")))
  #testRad = cbind(TestData[c(8:21)],TestData %>% select(matches("T2")))
  
  
  Rad = TrainData[c(22:8716)]
  testRad = TestData[c(22:8716)]
  
  
  
  x_noRad = as.matrix(noRad)
  fit_noRad = Coxnet(x_noRad, y, Omega = NULL, penalty = "Lasso", nfolds = 20)
  
  x_Rad = as.matrix(Rad)
  fit_Rad = Coxnet(x_Rad, y, Omega = NULL, penalty = "Lasso", nfolds = 20)
  

 
  
  vars_noRad = which(fit_noRad$Beta0!=0)
  colnames(noRad[,c(vars_noRad)])
  coefs_noRad = fit_noRad$Beta[c(which(fit_noRad$Beta0!=0))]
 
  vars_Rad = which(fit_Rad$Beta0!=0)
  colnames(Rad[,c(vars_Rad)])
  coefs_Rad = fit_Rad$Beta[c(which(fit_Rad$Beta0!=0))]
 
  
  
  TestData$pred_noRads = coefs_noRad[1]*testNoRad[,vars_noRad[1]]
  print( TestData$pred_noRads[1:3])
  for (i in 2:length(vars_noRad)){
    print(cat("i = ", i))
    print(colnames(testNoRad[vars_noRad[i]]))
    print(testNoRad[1:3,vars_noRad[i]])
    print(coefs_noRad[i])
    print(coefs_noRad[i]*testNoRad[1:3,vars_noRad[i]])
    TestData$pred_noRads = TestData$pred_noRads + coefs_noRad[i]*testNoRad[,vars_noRad[i]]
    print(TestData$pred_noRads[1:3])
    }
  
g1 = roc(TestData$status,TestData$pred_noRads)
auc1 = paste("Standard AUC = ",toString(round(g1$auc, digits = 3)))
best.coords <- coords(g1, "best", best.method="youden")


TestData$pred_Rads = coefs_Rad[1]*testRad[,vars_Rad[1]]
print(TestData$pred_Rads[1:3])
for (i in 2:length(vars_Rad)){
  print(cat("i = ", i))
  print(colnames(testRad[vars_Rad[i]]))
  print(TestData[1:3,vars_Rad[i]])
  print(coefs_Rad[i])
  print(coefs_Rad[i]*testRad[1:3,vars_Rad[i]])
  TestData$pred_Rads = TestData$pred_Rads + coefs_Rad[i]*testRad[,vars_Rad[i]]
  print(TestData$pred_Rads[1:3])
}

g2 = roc(TestData$status,TestData$pred_Rads)   
auc2 = paste("All Radiomics AUC = ",toString(round(g2$auc, digits = 3)))
best.coords <- coords(g2, "best", best.method="youden")





plot(g1, col = 'purple')
text(0.8,0.15,'--',  col = 'purple')
text(0.46,.15,auc1,  col = 'purple')

plot(g2, add=TRUE, col='#2166AC')
text(0.8,0.10,'--', col = '#2166AC')
text(0.45,0.10,auc2,col='#2166AC')


prop.test(statusCheck)
ci(g1)
ci(g2)
roc.test(g1, g2)



colnames(Rad[,c(vars_Rad)])
coefs_Rad
colnames(noRad[,c(vars_noRad)])
coefs_noRad




######################################################################
#################### Time-dependent ROC ############################
####################################################################

# we evaluate bilirubin as a prognostic biomarker.
ROC.Rads<-timeROC(T=TestData$time,
                  delta=TestData$status,marker=TestData$pred_Rads,
                  cause=1,weighting="marginal",
                  times=seq(5,75,5),
                  iid=TRUE)
# we evaluate albumin as a prognostic biomarker.
ROC.noRads<-timeROC(T=TestData$time,
                     delta=TestData$status,marker=TestData$pred_noRads,
                     cause=1,weighting="marginal",
                     times=seq(5,75,5),
                     iid=TRUE)
# we compare albumin and bilirubin as prognostic biomarkers.
ROC.Rads #print results for albumin
ROC.noRads #print results for bilirubin
compare(ROC.Rads,ROC.noRads) #compute p-values of comparison tests
compare(ROC.Rads,ROC.noRads,adjusted=TRUE)


plotAUCcurve(ROC.Rads,conf.int=FALSE,col="red")
plotAUCcurve(ROC.noRads,conf.int=FALSE,col="blue",add=TRUE)
legend("bottomright",c("All Radiomics","Standard"),col=c("red","blue"),lty=1,lwd=2)
  