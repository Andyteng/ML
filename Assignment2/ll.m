function [logL] = ll(X)
%  Useful info
labels = getlab(X);
X = getdata(X);
[N,D] = size(X);
X0 = X(labels == -1,:);
N0 = size(X0,1);
X1 = X(labels == 1,:);
N1 = size(X1,1);

prior0 = N0/(N0+N1);
prior1 = N1/(N0+N1);

mean0 = mean(X0,1);
mean1 = mean(X1,1);

sigma0 = cov(X0);
sigma1 = cov(X1);

sigma = prior0 * sigma0 + prior1 * sigma1 + 1e-10*eye(D);

logL = sum(log(mvnpdf(X(labels==-1,:), mean0, sigma)+eps(0)))+sum(log(mvnpdf(X(labels==1,:), mean1, sigma)+eps(0)));
end 