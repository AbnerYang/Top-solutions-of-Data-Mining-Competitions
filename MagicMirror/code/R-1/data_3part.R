train.gf.9w=read.csv('feature/train_x_gf.csv',sep = ",",header = TRUE)
train.gf.3part.9w=data.frame(cbind(train.gf.9w[,1],train.gf.9w[,91:209]))
rm(train.gf.9w)
# train.gf.3part.9w=train.gf.3part.9w[,selectFeaturesByVar(train.gf.3part.9w,0)]
# for(i in 2:ncol(train.gf.3part.9w)){
#   train.gf.3part.9w[which(train.gf.3part.9w[,i]==-1),i]=NaN
# }
# train.gf.3part.9w=fill_missing(train.gf.3part.9w)
train.gf.3part.9w[is.na(train.gf.3part.9w)]=-1
train.gf.3part.9w.result=data.frame(cal_3part(train.gf.3part.9w,0.1))
train.gf.3part.9w.result=cbind(train.gf.3part.9w[,1],train.gf.3part.9w.result)

train.3part.9w=train.gf.3part.9w.result
rm(train.gf.3part.9w.result)
train.8w.idx=read.csv('feature/train_x_gf_just_train.csv',sep = ",",header = TRUE)$Idx
train.8w.idx.Index=match(train.8w.idx,train.gf.3part.9w[,1])

train.3part.8w=data.frame(train.3part.9w[train.8w.idx.Index,])
test.3part.1w=data.frame(train.3part.9w[-train.8w.idx.Index,])
rm(train.gf.3part.9w)

daily.test=read.csv('daily_test.csv',sep = ",",header = TRUE)
final.test=read.csv('final_test.csv',sep = ",",header = TRUE)

val.idxIndex.2w=match(c(daily.test[,1],final.test[,1]),train.3part.8w[,1])
val.3part.2w=train.3part.8w[val.idxIndex.2w,]
train.3part.6w=train.3part.8w[-val.idxIndex.2w,]

train.3w.idx=read.csv('feature/round1_idx.csv',sep = ",",header = TRUE)$Idx
train.3w.idxIndex=match(train.3w.idx,train.3part.8w[,1])
train.3part.3w=data.frame(train.3part.8w[train.3w.idxIndex,])

call_lasso_cv(train.3part.8w,train.y.8w,3)
t=call_lasso(train.3part.6w,train.y.6w,val.3part.2w,val.y.2w,daily_test,final.test)

call_liblinear_cv(train.3part.8w,train.y.8w,3)
t=call_liblinear(train.3part.6w,train.y.6w,val.3part.2w,val.y.2w,daily_test,final.test)

call_svmlinear_cv(train.3part.8w,train.y.8w,3)
t=call_svmlinear(train.3part.6w,train.y.6w,val.3part.2w,val.y.2w,daily_test,final.test)

write.csv(train.3part.8w,file='train.3part.8w.csv',row.names=F,quote = F)
write.csv(test.3part.1w,file='test.3part.1w.csv',row.names=F,quote = F)

# train.waibu.8w=read.csv('feature/round2/third_party_features.csv',sep = ",",header = FALSE)
# train.waibu.8w[is.na(train.waibu.8w)]=-1
# call_lasso_cv(train.waibu.8w,train.y.8w,10)

# train.gf.3part.8w.summary=c()
# for(i in 2:8){
#   train.gf.3part.summary.8w.i=train.gf.3part.8w.result[,c(i,i+24,i+24*2,i+24*3,i+24*4,i+24*5,i+24*6,i+24*7,i+24*8,i+24*9,i+24*10,i+24*11,i+24*12,i+24*13,i+24*14,i+24*15,i+24*16)]  
#   summary.i.hasvalue=sapply(1:nrow(train.gf.3part.summary.8w.i),function(j)length(train.gf.3part.summary.8w.i[j,train.gf.3part.summary.8w.i[j,]>0]))
#   summary.i.zero=sapply(1:nrow(train.gf.3part.summary.8w.i),function(j)length(train.gf.3part.summary.8w.i[j,train.gf.3part.summary.8w.i[j,]==0]))
#   summary.i.sum=rowSums(train.gf.3part.summary.8w.i)
#   summary.i.avg=rowMeans(train.gf.3part.summary.8w.i)
#   summary.i.max=sapply(1:nrow(train.gf.3part.summary.8w.i),function(j)max(train.gf.3part.summary.8w.i[j,]))
#   summary.i.min.opt=sapply(1:nrow(train.gf.3part.summary.8w.i),function(j)min(train.gf.3part.summary.8w.i[j,]))
#   train.gf.3part.8w.summary=cbind(train.gf.3part.8w.summary,
#                                  summary.i.hasvalue,
#                                  summary.i.zero,
#                                  summary.i.sum,
#                                  summary.i.avg,
#                                  summary.i.max,
#                                  summary.i.min.opt)
# }
# train.gf.3part.8w.summary=cbind(train.gf.3part.8w.result[,1],train.gf.3part.8w.summary)