function [ train_new, unlabeled_new ] = update_dataset( index,train,unlabeled )

data = unlabeled.data;
lab = unlabeled.labels;
ele = prdataset(data(index,:),lab(index));
data(index,:) = [];
lab(index) = [];
unlabeled_new = prdataset(data,lab);

train_new = [train; ele];
end
