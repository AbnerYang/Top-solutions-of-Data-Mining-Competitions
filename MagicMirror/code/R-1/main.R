train.2445.3w=read.csv('feature/round2/train.x.2w.2445.csv',sep = ",",header = TRUE)
train.2445.3w=cbind(train.3w.idx,train.2445.3w)
# for(i in 2:ncol(train.2445.3w)){
#   train.2445.3w[which(train.2445.3w[,i]==-1),i]=NaN
# }
# train.2445.3w=fill_missing(train.2445.3w)
train.2445.3w[is.na(train.2445.3w)]=-1

index=read.csv('feature/round1_test.csv',sep = ",",header = TRUE)$Idx
val.2445.2w=read.csv('feature/round2/test.x.2w.2445.csv',sep = ",",header = TRUE)
val.2445.2w=cbind(index,val.2445.2w)
# for(i in 2:ncol(val.2445.2w)){
#   val.2445.2w[which(val.2445.2w[,i]==-1),i]=NaN
# }
# val.2445.2w=fill_missing(val.2445.2w)
val.2445.2w[is.na(val.2445.2w)]=-1
resort.index=match(c(daily.test[,1],final.test[,1]),val.2445.2w[,1])
val.2445.2w=val.2445.2w[resort.index,]

train.3w.v1=cbind(train.2445.3w,train.3part.3w[,-1])
val.2w.v1=cbind(val.2445.2w,val.3part.2w[,-1])

call_lasso_cv(train.3w.v1,train.y.3w,5)

pred.2w.lasso=call_lasso(train.3w.v1,train.y.3w,val.2w.v1,val.y.2w,daily_test,final.test)
pred.2w.liblinear=call_liblinear(train.3w.v1,train.y.3w,val.2w.v1,val.y.2w,daily_test,final.test)
pred.2w.svmlinear=call_svmlinear(train.3w.v1,train.y.3w,val.2w.v1,val.y.2w,daily_test,final.test)

cor(pred.2w.lasso[,2],pred.2w.svmlinear[,2])
cal_auc(val.y.2w,(4*pred.2w.lasso[,2]+1*pred.2w.liblinear[,2]+2*pred.2w.svmlinear[,2])/7)
cal_auc(val.y.2w,pred.2w.lasso[,2])#0.767374772501388
cal_auc(val.y.2w,pred.2w.liblinear[,2])#0.766127766654036
cal_auc(val.y.2w,pred.2w.svmlinear[,2])#0.766456945437347

res=cbind(pred.2w.lasso[,1],(4*pred.2w.lasso[,2]+1*pred.2w.liblinear[,2]+2*pred.2w.svmlinear[,2])/7)
res=res[order(as.character(res[,1])),]
write.csv(res,file='ensemble.linear.2w.7680.csv',row.names=F,quote = F)
############################################################################################
train.2485.8w=read.csv('feature/round2/train.x.8w.2485.csv',sep = ",",header = TRUE)
train.2485.8w=data.frame("idx"=train.3part.8w[,1],train.2485.8w)
test.2485.1w=read.csv('feature/round2/test.x.8w.2485.csv',sep = ",",header = TRUE)
test.2485.1w=data.frame("idx"=test.3part.1w[,1],test.2485.1w)

full.2485.9w=rbind(train.2485.8w,test.2485.1w)
for(i in 2:ncol(full.2485.9w)){
  full.2485.9w[which(full.2485.9w[,i]==-1),i]=NaN
}
full.2485.9w=fill_missing(full.2485.9w)
full.2485.9w[is.na(full.2485.9w)]=-1
train.8w.v1=cbind(full.2485.9w[train.8w.idx.Index,],train.3part.8w[,-1])
test.1w.v1=cbind(full.2485.9w[-train.8w.idx.Index,],test.3part.1w[,-1])
rm(full.2485.9w)
rm(train.2485.8w)
write.csv(train.8w.v1,file='train.8w.v1.csv',row.names=F,quote = F)
write.csv(test.1w.v1,file='test.1w.v1.csv',row.names=F,quote = F)
############################################################################################
train_x=read.csv('feature/round2/train.8w.v1.csv',sep = ",",header = TRUE)
test_x=read.csv('feature/round2/test.1w.v1.csv',sep = ",",header = TRUE)

train_x=data.frame("idx"=train_x[,1],train_x[,-1])
test_x=data.frame("idx"=test_x[,1],test_x[,-1])
full.x=rbind(data.frame("seq"=0,train_x[,-1]),data.frame("seq"=1,test_x[,-1]))

full.x.1=full.x[,1:1500]
full.x.2=full.x[,1501:2673]
full.x.1=data.frame(apply(full.x.1,2,normalize))
full.x.2=data.frame(apply(full.x.2,2,normalize))
full.x=cbind(full.x.1,full.x.2)
full.x[is.na(full.x)]=-1

train_x=data.frame("idx"=train_x[,1],full.x[full.x$seq==0,][,-1])
test_x=data.frame("idx"=test_x[,1],full.x[full.x$seq==1,][,-1])

train.8w.v1=train_x
test.1w.v1=test_x

write.csv(train.8w.v1,file='train.8w.v1.final.csv',row.names=F,quote = F)
write.csv(test.1w.v1,file='test.1w.v1.final.csv',row.names=F,quote = F)
############################################################################################
train.8w.v1=read.csv('feature/round2/train.8w.v1.final.csv',sep = ",",header = TRUE)
test.1w.v1=read.csv('feature/round2/test.1w.v1.final.csv',sep = ",",header = TRUE)

set.seed(76)
cv.fit <- cv.glmnet(data.matrix(train.8w.v1[,-1]), train.y.8w, family = "binomial",
                    nfolds=5,type.measure="auc",type.logistic="Newton",
                    alpha=1,dfmax=550,standardize=FALSE)
print(paste0('cv max:',max(cv.fit$cvm)))#7745


test.lasso.final=as.numeric(predict(cv.fit,as.matrix(test.1w.v1[,-1]), s=cv.fit$lambda.min,type="response")[,1])
test.lasso.final=cbind(test.1w.v1[,1],test.lasso.final)

write.csv(data.frame('Idx'=test.lasso.final[,1],
                     'score'=test.lasso.final[,2]),
          file='test.lasso.result7745.csv',row.names=F,quote = F)

call_liblinear_cv(train.8w.v1,train.y.8w,10)#7756

call_svmlinear_cv(train.8w.v1,train.y.8w,10)#7734

set.seed(76)
model.liblinear=LiblineaR(data=data.matrix(train.8w.v1[,-1]),target=as.factor(train.y.8w),type=6,cost=0.5,verbose=2,epsilon=0.001)#0.0008
test.liblinear.final=predict(model.liblinear,test.1w.v1[,-1],proba=T)$probabilities[,2]
test.liblinear.final=cbind(test.1w.v1[,1],test.liblinear.final)


model.svmlinear=LiblineaR(data=data.matrix(train.8w.v1[,-1]),target=as.factor(train.y.8w),type=1,cost=0.005,verbose=2,epsilon=0.01)
test.svmlinear.final=predict(model.svmlinear,test.1w.v1[,-1],proba=F,decisionValues=T)$decisionValues[,1]
test.svmlinear.final=as.numeric(normalizeData(-test.svmlinear.final,type="0_1"))
test.svmlinear.final=cbind(test.1w.v1[,1],test.svmlinear.final)



test.lasso.result7745=read.csv('test.lasso.result7745.csv',sep = ",",header = TRUE)
cor(test.lasso.result7745[,2],test.svmlinear.final[,2])

final.res=cbind(test.lasso.result7745[,1],(4*test.lasso.result7745[,2]+1*test.liblinear.final[,2]+2*test.svmlinear.final[,2])/7)

test.lasso.result7788=read.csv('test.lasso.result7788.csv',sep = ",",header = TRUE)
cor(final.res[,2],test.lasso.result7788[,2])

write.csv(data.frame('Idx'=final.res[,1],
                     'score'=final.res[,2]),
          file='test.1w.ensemble20160418.csv',row.names=F,quote = TRUE)

############################################################################################
svmlinear.2w.7483=read.csv('svmlinear.2w.7483.csv',sep = ",",header = TRUE)
liblinear.2w.7594=read.csv('liblinear.2w.7594.csv',sep = ",",header = TRUE)
lasso.2w.7618=read.csv('lasso.2w.7618.csv',sep = ",",header = TRUE)
svmlinear.2w.7483=svmlinear.2w.7483[resort.index,]
liblinear.2w.7594=liblinear.2w.7594[resort.index,]
lasso.2w.7618=lasso.2w.7618[resort.index,]
cal_auc(val.y.2w,0.8*lasso.2w.7618[,2]+0.1*liblinear.2w.7594[,2]+0.1*svmlinear.2w.7483[,2])
res=cbind(lasso.2w.7618[,1],0.8*lasso.2w.7618[,2]+0.1*liblinear.2w.7594[,2]+0.1*svmlinear.2w.7483[,2])
res=res[order(as.character(res[,1])),]
write.csv(res,file='ensemble.linear.2w.7631.csv',row.names=F,quote = F)
