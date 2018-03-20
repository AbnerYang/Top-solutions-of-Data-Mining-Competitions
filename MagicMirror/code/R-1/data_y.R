daily.test=read.csv('daily_test.csv',sep = ",",header = TRUE)
final.test=read.csv('final_test.csv',sep = ",",header = TRUE)

train.y=read.csv('feature/round2/idx.target.csv',sep = ",",header = TRUE)
train.y.8w=train.y$target

val.y.2w=train.y[val.idxIndex.2w,]$target
train.y.6w=train.y[-val.idxIndex.2w,]$target
train.y.3w=train.y[train.3w.idxIndex,]$target

rm(train.y)

#train.3w.idx=read.csv('round1_idx.csv',sep = ",",header = TRUE)$Idx
# train.idxIndex.3w=match(train.3w.idx,train.new.8w[,1])
# train.new.3w=train.new.8w[train.idxIndex.3w,]
# train.y.3w=train.y[train.idxIndex.3w,]$target