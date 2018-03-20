library(xgboost)
library(caret)
library(RSNNS)

library(LiblineaR)
library(glmnet)
library(kernlab)
library(kknn)
library(party)
library(ROCR)
library(caretEnsemble)
library(InformationValue)
library(sqldf)
library(tcltk)
library(matrixStats)
cal_auc=function(act,pred){
  rocr.perf=performance(
    prediction(pred,act),'auc')
  as.numeric(rocr.perf@y.values)
}
Mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- "-999"
  return(xmode)
}

call_lasso=function(train_x,train_y,test_x,test_y,daily_test,final.test){
  a=Sys.time()
  
  train_x=data.frame("idx"=train_x[,1],train_x[,-1])
  test_x=data.frame("idx"=test_x[,1],test_x[,-1])
  full.x=rbind(data.frame("seq"=0,train_x[,-1]),data.frame("seq"=1,test_x[,-1]))
  full.x=data.frame(apply(full.x,2,normalize))
  full.x[is.na(full.x)]=-1
  train_x=data.frame("idx"=train_x[,1],full.x[full.x$seq==0,][,-1])
  test_x=data.frame("idx"=test_x[,1],full.x[full.x$seq==1,][,-1])
  
  set.seed(76)
  cv.fit <- cv.glmnet(data.matrix(train_x[,-1]), train_y, family = "binomial",
                      nfolds=5,type.measure="auc",type.logistic="Newton",
                      alpha=1,dfmax=550,standardize=FALSE)
  print(paste0('cv max:',max(cv.fit$cvm)))
  test.lasso=as.numeric(predict(cv.fit,as.matrix(test_x[,-1]), s=cv.fit$lambda.min,type="response")[,1])
  test.lasso.result=cbind(test_x[,1],test.lasso)
  dailyIndex=match(daily.test[,1],test.lasso.result[,1])
  finalIndex=match(final.test[,1],test.lasso.result[,1])
  print(paste0('average:',cal_auc(test_y,test.lasso)))
  print(paste0('daily_test:',cal_auc(daily.test[,2],test.lasso.result[dailyIndex,2])))
  print(paste0('final_test:',cal_auc(final.test[,2],test.lasso.result[finalIndex,2])))
  b=Sys.time()
  print(b-a)
  return (test.lasso.result)
}
call_lasso_cv=function(train.x,train.y,k){
  a=Sys.time()
  train.x=data.frame(apply(train.x[,-1],2,normalize))
  train.x[is.na(train.x)]=-1
  set.seed(76)
  cv.fit <- cv.glmnet(data.matrix(train.x), train.y, family = "binomial",
                      nfolds=k,type.measure="auc",type.logistic="Newton",
                      alpha=1,dfmax=550,standardize=FALSE)
  print(max(cv.fit$cvm))
  b=Sys.time()
  print(b-a)
}
call_liblinear=function(train_x,train_y,test_x,test_y,daily_test,final.test){
  a=Sys.time()
  train_x=data.frame("idx"=train_x[,1],train_x[,-1])
  test_x=data.frame("idx"=test_x[,1],test_x[,-1])
  full.x=rbind(data.frame("seq"=0,train_x[,-1]),data.frame("seq"=1,test_x[,-1]))
  full.x=data.frame(apply(full.x,2,normalize))
  full.x[is.na(full.x)]=-1
  train_x=data.frame("idx"=train_x[,1],full.x[full.x$seq==0,][,-1])
  test_x=data.frame("idx"=test_x[,1],full.x[full.x$seq==1,][,-1])
  set.seed(76)
  model=LiblineaR(data=data.matrix(train_x[,-1]),target=as.factor(train_y),type=6,cost=0.5,verbose=2,epsilon=0.001)#0.0008
  test.liblinear=predict(model,test_x[,-1],proba=T)$probabilities[,2]
  test.liblinear.result=cbind(test_x[,1],test.liblinear)
  dailyIndex=match(daily.test[,1],test.liblinear.result[,1])
  finalIndex=match(final.test[,1],test.liblinear.result[,1])
  print(paste0('average:',cal_auc(test_y,test.liblinear)))
  print(paste0('daily_test:',cal_auc(daily.test[,2],test.liblinear.result[dailyIndex,2])))
  print(paste0('final_test:',cal_auc(final.test[,2],test.liblinear.result[finalIndex,2])))
  b=Sys.time()
  print(b-a)
  return(test.liblinear.result)
}
call_liblinear_cv=function(train.x,train.y,nfolds){
  a=Sys.time()
  auclist=c()
  #train.x=data.frame(apply(train.x[,-1],2,normalize))
  #train.x[is.na(train.x)]=-1
  train.x=train.x[,-1]
  set.seed(76)
  train.folds=createFolds(train.y,k=nfolds,list=F)
  for(i in 1:nfolds){
    train.x.i=train.x[train.folds!=i,]
    train.y.i=train.y[train.folds!=i]
    val.x.i=train.x[train.folds==i,]
    val.y.i=train.y[train.folds==i]
    model=LiblineaR(data=data.matrix(train.x.i),target=as.factor(train.y.i),type=6,cost=0.5,verbose=2,epsilon=0.001)
    test.liblinear=predict(model,val.x.i,proba=T)$probabilities[,2]
    auclist=c(auclist,cal_auc(val.y.i,test.liblinear))
  }
  print(mean(unlist(auclist)))
  b=Sys.time()
  print(b-a)
}
call_svmlinear=function(train_x,train_y,test_x,test_y,daily_test,final.test){
  a=Sys.time()
  set.seed(76)
  train_x=data.frame("idx"=train_x[,1],train_x[,-1])
  test_x=data.frame("idx"=test_x[,1],test_x[,-1])
  full.x=rbind(data.frame("seq"=0,train_x[,-1]),data.frame("seq"=1,test_x[,-1]))
  full.x=data.frame(apply(full.x,2,normalize))
  full.x[is.na(full.x)]=-1
  train_x=data.frame("idx"=train_x[,1],full.x[full.x$seq==0,][,-1])
  test_x=data.frame("idx"=test_x[,1],full.x[full.x$seq==1,][,-1])
  
  model=LiblineaR(data=data.matrix(train_x[,-1]),target=as.factor(train_y),type=1,cost=0.005,verbose=2,epsilon=0.01)
  test.svmlinear=predict(model,test_x[,-1],proba=F,decisionValues=T)$decisionValues[,1]
  test.svmlinear=as.numeric(normalizeData(-test.svmlinear,type="0_1"))
  test.svmlinear.result=cbind(test_x[,1],test.svmlinear)
  dailyIndex=match(daily.test[,1],test.svmlinear.result[,1])
  finalIndex=match(final.test[,1],test.svmlinear.result[,1])
  print(paste0('average:',cal_auc(test_y,test.svmlinear)))
  print(paste0('daily_test:',cal_auc(daily.test[,2],test.svmlinear.result[dailyIndex,2])))
  print(paste0('final_test:',cal_auc(final.test[,2],test.svmlinear.result[finalIndex,2])))
  b=Sys.time()
  print(b-a)
  return(test.svmlinear.result)
}
call_svmlinear_cv=function(train.x,train.y,nfolds){
  a=Sys.time()
  auclist=c()
#   train.x=data.frame(apply(train.x[,-1],2,normalize))
#   train.x[is.na(train.x)]=-1
  train.x=train.x[,-1]
  set.seed(76)
  train.folds=createFolds(train.y,k=nfolds,list=F)
  for(i in 1:nfolds){
    train.x.i=train.x[train.folds!=i,]
    train.y.i=train.y[train.folds!=i]
    val.x.i=train.x[train.folds==i,]
    val.y.i=train.y[train.folds==i]
  
    model=LiblineaR(data=data.matrix(train.x.i),target=as.factor(train.y.i),type=1,cost=0.005,verbose=2,epsilon=0.01)
    test.svmlinear=predict(model,val.x.i,proba=F,decisionValues=T)$decisionValues[,1]
    test.svmlinear=as.numeric(normalizeData(-test.svmlinear,type="0_1"))
    auclist=c(auclist,cal_auc(val.y.i,test.svmlinear))
  }
  print(mean(unlist(auclist)))
  b=Sys.time()
  print(b-a)
}
fill_missing=function(train.new){
  for (var in 1:ncol(train.new)) {
    if (class(train.new[,var])%in% c("integer", "numeric")) {
      if(length(train.new[is.na(train.new[,var]),var])/length(train.new[,var])<0.25){
        train.new[is.na(train.new[,var]),var] <- mean(train.new[,var], na.rm = TRUE) 
      }
    } else if (class(train.new[,var]) %in% c("character", "factor")) {
      if(length(train.new[is.na(train.new[,var]),var])/length(train.new[,var])<0.25){
        train.new[is.na(train.new[,var]),var] <- Mode(train.new[,var], na.rm = TRUE)
      }
    }
  }
  return(train.new)
}


cal_3part=function(train.gf.3part.8w,C){
  train.gf.3part.8w.v.i=c()
  train.gf.3part.8w.result=c()
  for(i in 1:17){
    train.gf.3part.8w.v.i=cbind(train.gf.3part.8w[,paste0("ThirdParty_Info_Period1_",i)],
                                train.gf.3part.8w[,paste0("ThirdParty_Info_Period2_",i)],
                                train.gf.3part.8w[,paste0("ThirdParty_Info_Period3_",i)],
                                train.gf.3part.8w[,paste0("ThirdParty_Info_Period4_",i)],
                                train.gf.3part.8w[,paste0("ThirdParty_Info_Period5_",i)],
                                train.gf.3part.8w[,paste0("ThirdParty_Info_Period6_",i)],
                                train.gf.3part.8w[,paste0("ThirdParty_Info_Period7_",i)])
    part.i.hasvalue=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)length(train.gf.3part.8w.v.i[j,train.gf.3part.8w.v.i[j,]>0]))
    part.i.missing=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)length(train.gf.3part.8w.v.i[j,train.gf.3part.8w.v.i[j,]==-1]))
    part.i.zero=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)length(train.gf.3part.8w.v.i[j,train.gf.3part.8w.v.i[j,]==0]))
    part.i.sum=rowSums(train.gf.3part.8w.v.i)
    part.i.avg=rowMedians(train.gf.3part.8w.v.i)
    part.i.max=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)max(train.gf.3part.8w.v.i[j,]))
    #part.i.min=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)min(train.gf.3part.8w.v.i[j,]))
    part.i.min.opt=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)min_opt_row(train.gf.3part.8w.v.i[j,]))
    
    part.i.sumbin=evenbins(part.i.sum,100)
    part.i.avgbin=evenbins(part.i.avg,100)
    part.i.maxbin=evenbins(part.i.max,100)
    part.i.minbin=evenbins(part.i.min.opt,100)
    
#     part.i.slope1=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_ra_rb(train.gf.3part.8w.v.i[j,2],train.gf.3part.8w.v.i[j,1]))
#     part.i.slope2=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_ra_rb(train.gf.3part.8w.v.i[j,3],train.gf.3part.8w.v.i[j,2]))
#     part.i.slope3=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_ra_rb(train.gf.3part.8w.v.i[j,4],train.gf.3part.8w.v.i[j,3]))
#     part.i.slope4=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_ra_rb(train.gf.3part.8w.v.i[j,5],train.gf.3part.8w.v.i[j,4]))
#     part.i.slope5=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_ra_rb(train.gf.3part.8w.v.i[j,6],train.gf.3part.8w.v.i[j,5]))
#     part.i.slope6=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_ra_rb(train.gf.3part.8w.v.i[j,7],train.gf.3part.8w.v.i[j,6]))
#     
#     part.i.slope1bin=evenbins(part.i.slope1,100)
#     part.i.slope2bin=evenbins(part.i.slope2,100)
#     part.i.slope3bin=evenbins(part.i.slope3,100)
#     part.i.slope4bin=evenbins(part.i.slope4,100)
#     part.i.slope5bin=evenbins(part.i.slope5,100)
#     part.i.slope6bin=evenbins(part.i.slope6,100)
#     part.i.slope7=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_va_vb(train.gf.3part.8w.v.i[j,3:4],train.gf.3part.8w.v.i[j,1:2]))
#     part.i.slope8=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_va_vb(train.gf.3part.8w.v.i[j,4:5],train.gf.3part.8w.v.i[j,2:3]))
#     part.i.slope9=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_va_vb(train.gf.3part.8w.v.i[j,5:6],train.gf.3part.8w.v.i[j,3:4]))
#     part.i.slope10=sapply(1:nrow(train.gf.3part.8w.v.i),function(j)divide_opt_va_vb(train.gf.3part.8w.v.i[j,6:7],train.gf.3part.8w.v.i[j,4:5]))
#     
#     part.i.slope11=(train.gf.3part.8w.v.i[,2]+train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4])/(train.gf.3part.8w.v.i[,1]+train.gf.3part.8w.v.i[,2]+train.gf.3part.8w.v.i[,3]+C)-1
#     part.i.slope12=(train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5])/(train.gf.3part.8w.v.i[,2]+train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+C)-1
#     part.i.slope13=(train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5]+train.gf.3part.8w.v.i[,6])/(train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5]+C)-1
#     part.i.slope14=(train.gf.3part.8w.v.i[,5]+train.gf.3part.8w.v.i[,6]+train.gf.3part.8w.v.i[,7])/(train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5]+train.gf.3part.8w.v.i[,6]+C)-1
#     
#     part.i.slope15=(train.gf.3part.8w.v.i[,2]+train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5])/(train.gf.3part.8w.v.i[,1]+train.gf.3part.8w.v.i[,2]+train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+C)-1
#     part.i.slope16=(train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5]+train.gf.3part.8w.v.i[,6])/(train.gf.3part.8w.v.i[,2]+train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5]+C)-1
#     part.i.slope17=(train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5]+train.gf.3part.8w.v.i[,6]+train.gf.3part.8w.v.i[,7])/(train.gf.3part.8w.v.i[,3]+train.gf.3part.8w.v.i[,4]+train.gf.3part.8w.v.i[,5]+train.gf.3part.8w.v.i[,6]+C)-1
#     
    train.gf.3part.8w.result=cbind(train.gf.3part.8w.result,
                                   part.i.hasvalue,part.i.missing,part.i.zero,
                                   part.i.sum,
                                   part.i.avg,
                                   part.i.max,
                                   #part.i.min,
                                   part.i.min.opt,
                                   part.i.sumbin,
                                   part.i.avgbin,
                                   part.i.maxbin,
                                   part.i.minbin)
#                                    part.i.slope1bin,
#                                    part.i.slope2bin,
#                                    part.i.slope3bin,
#                                    part.i.slope4bin,
#                                    part.i.slope5bin,
#                                    part.i.slope6bin,
#                                    part.i.slope1,
#                                    part.i.slope2,
#                                    part.i.slope3,
#                                    part.i.slope4,
#                                    part.i.slope5,
#                                    part.i.slope6)
                                   #part.i.slope7,
                                   #part.i.slope8,
                                   #part.i.slope9,
                                   #part.i.slope10)
                                   #part.i.slope11,part.i.slope12,part.i.slope13,
                                   #part.i.slope14,part.i.slope15,part.i.slope16,part.i.slope17)
  }
  return (train.gf.3part.8w.result)
}
min_opt_row=function(row_vector){
  min=c()
  if(length(row_vector[row_vector==-1])==length(row_vector)){
    min=-1
  }else{
    min=min(row_vector[row_vector>=0])
  }
  return (min)
}
divide_opt_ra_rb=function(a,b){
  rate=c()
  if(b==0){
    rate=a-1
  }else{
    rate=a/b-1
  }
  return (rate)
}
divide_opt_va_vb=function(a,b){
  a1=length(a[a>0])
  a=sum(a[a>0])
  mean_a=c()
  if(a1==0){
    mean_a=a
  }else{
    mean_a=a/a1
  }
  b1=length(b[b>0])
  b=sum(b[b>0])
  if(b1==0){
    mean_b=b
  }else{
    mean_b=b/b1
  }
  if(mean_b==0){
    rate=mean_a-1
  }else{
    rate=mean_a/mean_b-1
  }
  return (rate)
}
evenbins <- function(x, bin.count=10, order=T) {
  bin.size <- rep(length(x) %/% bin.count, bin.count)
  bin.size <- bin.size + ifelse(1:bin.count <= length(x) %% bin.count, 1, 0)
  bin <- rep(1:bin.count, bin.size)
  if(order) {    
    bin <- bin[rank(x,ties.method="random")]
  }
  return(factor(bin, levels=1:bin.count, ordered=order))
}

selectFeaturesByVar=function(train.x,C){
  selFeatureByVar=c()
  for(i in 1:ncol(train.x)){
    if(var(train.x[,i],na.rm=TRUE)>C){
      selFeatureByVar=c(selFeatureByVar,i)
    }
  }
  return (selFeatureByVar)
}

normalize<-function(m){
  (m - min(m))/(max(m)-min(m))
}

