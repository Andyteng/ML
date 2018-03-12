function [w, xi, b] = SVM_S(X_labeled,X_unlabeled,Cl,Cu)
m = size((X_unlabeled.data),1);
[l,dim] = size(X_labeled.data);
    cvx_begin
        variables w(dim) xi(l+m) b        % optim variables
        minimize( 0.5 * (w' * w) + Cl * sum(xi(1:l,1))+Cu * sum(xi(l+1:l+m,1)))      % objective
        subject to 
        X_labeled.labels .* (X_labeled.data * w + b) -1 + xi(1:l,1) >= 0;   % constraints
        X_unlabeled.labels.*(X_unlabeled.data * w + b )-1 + xi(l+1:l+m,1) >= 0;
        xi >= 0;
    cvx_end