function [X,rest] = gendata_split(num,X_origin)

    index = randperm(size(X_origin.data,1),num);
    data = X_origin.data(index,:);
    labels = X_origin.labels(index);
    
    X = prdataset(data, labels);
    
    index_u = 1:size(X_origin.data,1);
    index_u(index)=[];
    rest = X_origin(index_u,:);