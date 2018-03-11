function [ train_new, unlabeled_new ] = update_dataset( index,train,unlabeled )

unlabeled(index,:) = [];
unlabeled_new = unlabeled;

train_new = [train unlabeled(index,:)];
end
