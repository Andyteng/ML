%% using CVX solver 
clc
clear all
%% Load the MAGIC Gamma Telescope Data Set
load magic04.mat ; %Load input data file
load labels.mat;
index_g = find([g{:}] == 'g');
index_h = find([g{:}] == 'h');
labels = zeros(size(magic04,1),1);
labels(index_g) = 1;
labels(index_h) = -1;
X = ones(size(magic04,1),10);
for i = 1:10
    X(:,i) = magic04(:,i)./std(magic04(:,i)); 
end
X_dataset = prdataset(X,labels);

%% SVM initialization
% get useful info
Cl = 10000;     % set C
Cu = 1;
u_num = [0, 10, 20, 40, 80, 160, 320, 640, 1280]; 
[X_labeled, rest] = gendata_split(50,X_dataset); 
[X_unlabeled,X_test] = gendata_split(u_num(7),rest);
X_train = X_labeled.data;
labels_train = X_labeled.labels;
labels_test = X_unlabeled.labels;
X_test = X_unlabeled.data;
% load data
%load('linear_svm.mat');
% get useful info
[n, dim] = size(X_train);
[m, dim_u] = size(X_unlabeled);
% unlabeled dataset
X_dataset = prdataset(X_test,labels_test);
[X_unlabel, rest] = gendata_split(320,X_dataset);
%form the optimization problem
% C = 0.1;     % set C
% cvx_begin
%     variables w(dim) xi(n) b        % optim variables
%     minimize( 0.5 * (w' * w) + C * sum(xi))      % objective
%     subject to 
%     labels_train .* (X_train * w + b) -1 + xi >= 0;   % constraints
%     xi >= 0;
% cvx_end


Cl = 10000;     % set C
Cu = 4000;
cvx_begin
    variables w(dim) xi(n+m) b        % optim variables
    minimize( 0.5 * (w' * w) + Cl * sum(xi(1:n))+Cu * sum(xi(n+1:end)))      % objective
    subject to 
    labels_train .* (X_train * w + b) -1 + xi(1:n) >= 0;   % constraints
    X_unlabeled.labels.*(X_unlabeled.data*w + b) - 1 + xi(n+1:end) >= 0;
    xi >= 0;
cvx_end

%% Visualization of SVM classifier
class1 = X_train(labels_train==1,:);        % 
class2 = X_train(labels_train==-1,:);
figure;
plot(class1(:,1),class1(:,2), '+r');
hold on
plot(class2(:,1),class2(:,2), '+b');

y = linspace(min(X_train(:,2)), max(X_train(:,2)));
x = (-b - w(2)*y)/w(1);
x1= (1-b - w(2)*y)/w(1);
x2 = (-1-b - w(2)*y)/w(1);
plot(x,y);
plot(x1,y,'--');
plot(x2,y,'--');
title('SVM classifier (C = 10)');
legend('label 1', 'label -1');
hold off
%%
predicted_label = sign( X_train * w + b);
sum(predicted_label == labels_train) / 100

%%
predicted_label = sign( X_test * w + b);
sum(predicted_label == labels_test) / 900

%% low-complexity algorithm
w0 = randn(size(X_train,2),1);
b0 = randn(1);
C = 10000;
lambda = 1/(2*C);
w = w0;
b = b0;
%loss = sum( abs(1 - labels_train .* (X_train * w + b))) + lambda * norm(w);
T = 2000;
alpha = 0.006;
best_loss = 1000;
labels_train = X_labeled.labels;
X_train = X_labeled.data;
t0 = cputime;
for i = 1: T
    % compute loss and (sub)gradient
   % alpha = alpha0 / sqrt(i);
    margin =  (1 - labels_train .* (X_train * w + b));
    hinge_loss = (margin>0) .* margin;
    loss =sum( hinge_loss) + lambda * (norm(w)^2);
    %break rule
    if loss < best_loss
        best_loss  = loss; 
        best_w = w;
        best_b = b;
    end
    dw = - sum(labels_train .* X_train .* (margin>0) )'+ 2 * lambda * w;
    db = - sum(labels_train .* (margin>0));
    if norm(dw) <=1e-3 && norm(db) <=1e-3
        disp(i)
        break
    end
    %update
    w = w - alpha * dw;
    b = b - alpha * db;
end
t = cputime - t0

% original problem
margin =  (1 - labels_train .* (X_train * best_w + best_b));
xi = (margin>0) .* margin;
optimal_value = 0.5 * (best_w' * best_w) + C * sum(xi)

predicted_label = sign( X_train * best_w + best_b);
sum(predicted_label == labels_train) / 100

predicted_label = sign( X_train * w + b);
sum(predicted_label == labels_train) / 100

%%
plot(class1(:,1),class1(:,2), '+r');
hold on
plot(class2(:,1),class2(:,2), '+b');
hold on

y = linspace(min(X_train(:,2)), max(X_train(:,2)));
x = (-best_b - best_w(2)*y)/best_w(1);
x1= (1-best_b - best_w(2)*y)/best_w(1);
x2 = (-1-best_b - best_w(2)*y)/best_w(1);
plot(x,y);
plot(x1,y,'--');
plot(x2,y,'--');