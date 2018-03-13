function [w, xi, b] = SVM_S(X_labeled,X_unlabeled,Cl,Cu)
[n,dim] = size(X_labeled);
[m,~] = size(X_unlabeled);
    cvx_begin
        variables w(dim) xi(n+m) b        % optim variables
        minimize( 0.5 * (w' * w) + Cl * sum(xi(1:n))+Cu * sum(xi(n+1:end)))      % objective
        subject to 
        X_labeled.labels .* (X_labeled.data * w + b) -1 + xi(1:n) >= 0;   % constraints
        X_unlabeled.labels.*(X_unlabeled.data * w + b )-1 + xi(n+1:end) >= 0;
        xi >= 0;
    cvx_end