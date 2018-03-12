%% using CVX solver 
% load data
load('linear_svm.mat');
% get useful info
[n, dim] = size(X_train);

%form the optimization problem
% C = 10;     % set C
% cvx_begin
%     variables w(dim) xi(n) b        % optim variables
%     minimize( 0.5 * (w' * w) + C * sum(xi))      % objective
%     subject to 
%     labels_train .* (X_train * w + b) -1 + xi >= 0;   % constraints
%     xi >= 0;
% cvx_end
Cl = 1;
Cu = 0;
l = 100;
X = prdataset(X_train, labels_train);
[w,xi,b] = SVM_S(X,Cl,Cu,l);

y_hat = sign(X_test*w+b);


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
w0 = randn(2,1);
b0 = randn(1);
C = 10;
lambda = 1/(2*C);
w = w0;
b = b0;
%loss = sum( abs(1 - labels_train .* (X_train * w + b))) + lambda * norm(w);
T = 2000;
alpha = 0.006;
best_loss = 1000;

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