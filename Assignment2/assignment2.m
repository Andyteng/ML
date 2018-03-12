clc
clear all
close all

%% Load the MAGIC Gamma Telescope Data Set
load magic04.mat ; %Load input data file
load labels.mat;
index_g = find([g{:}] == 'g');
index_h = find([g{:}] == 'h');
labels = zeros(size(magic04,1),1);
labels(index_g) = 1;
labels(index_h) = 0;
X = ones(size(magic04,1),10);
for i = 1:10
    X(:,i) = magic04(:,i)./std(magic04(:,i)); 
end
X_dataset = prdataset(X,labels);

%% basic LDA testing
[X_labeled, rest] = gendata_split(25,X_dataset); 
X = X_labeled;
[mean0, mean1, sigma,prior0] = llda(X,25, 500);
[e, ll] = llda_predict(rest,mean0,mean1,sigma,prior0);

%%
repeats = 50;
e_history = zeros(9,repeats);
LL = zeros(9,repeats);
e_history1 = zeros(9,repeats);
LL1 = zeros(9,repeats);
u_num = [0, 10, 20, 40, 80, 160, 320, 640, 1280]; 

for j=1:repeats
    [X_labeled, rest] = gendata_split(25,X_dataset); 
    for i = 1:9
        [X_unlabeled,X_test] = gendata_split(u_num(i),rest);
        X = [X_labeled; X_unlabeled];
        [mean0, mean1, sigma,prior0] = lda(X, 25, 300);
        [e,ll]= lda_predict(X_test,mean0,mean1,sigma,prior0);
        e_history(i,j) = e; 
        LL(i,j) = ll;
    end
end
plot(u_num,mean(e_history,2));
legend('expected error rates')
ylabel('averaged error(50 experiments)');
xlabel('num. of added unlabeled samples');
figure;
plot(u_num,mean(LL,2));
legend('log likelihood')
ylabel('averaged loglikelihood(50 experiments)');
xlabel('num. of added unlabeled samples');

%% EM

u_num = [0,10,20,40,80,160,320,640];
n_repeat = 5;
e = zeros(n_repeat,length(u_num)-1);
e_sv = zeros(n_repeat,length(u_num));
for r = 1:n_repeat
    err = [];
    % labeled dataset
    [X_labeled, rest] = gendata(25,X_dataset); 
    label_l = X_labeled.labels;
    train_l = X_labeled.data;

    class1_index = find(label_l == 1);
    class2_index = find(label_l == 0);
    class1 = X_labeled(class1_index,:);
    class2 = X_labeled(class2_index,:);

    %supervised error rate
    err_supervised = X_dataset*ldc(X_labeled)*testc;
    u_num = [0,10,20,40,80,160,320,640];
    for i = 2:length(u_num)
        % unlabeled dataset
        [X_unlabeled,rrest] = gendata(u_num(i),rest);
        label_u = X_unlabeled.labels;
        train_u = X_unlabeled.data;
        err_num = 0;
        count = 0;
        y_hat = zeros(size(label_u,1),1);
        %fprintf("%i unlabeled samples are added into the training dataset\n", u_num(i))
        while size(X_unlabeled.data,1) ~= 0
            %fprintf("%i samples were labeled \n", count);
            [mu1_hat, s1_hat] = gaussian_ML_estimate(class1.data');  
            [mu2_hat, s2_hat] = gaussian_ML_estimate(class2.data'); 
            mu_hat = [mu1_hat, mu2_hat];  
            s_hat = (1/2) * (s1_hat + s2_hat); 
            [v_min, c, i_min] = euclidean_classifier(mu_hat, X_unlabeled.data');
            if c == 1
                [class1, X_unlabeled] = update_dataset(i_min,class1,X_unlabeled);
                y_hat(i_min+count) = 1;
                count= count+1;
                %fprintf('sample %i is added to class1\n', i_min);
                if label_u(i_min) ~= 1
                    err_num = err_num+1;
                end
            else 
                [class2, X_unlabeled] = update_dataset(i_min,class2,X_unlabeled);
                y_hat(i_min) = 1;
                count= count+1;
                %fprintf('sample %i is added to class2\n', i_min);
                if label_u(i_min) ~= 0
                    err_num = err_num+1;
                end
            end
        end
        fprintf('with error number: %i \n', err_num)
        disp('all unlabeled data were labeled.')
        X_uunlabeled = prdataset(train_u,y_hat);
        X_llabeled = [X_labeled;X_uunlabeled];
        error = X_dataset*fisherc(X_llabeled)*testc;
        err = [err; error];
    end
    e(r,:) = err;
end
plot(u_num(2:end),sum(e)./n_repeat);
hold on
plot(u_num(2:end),sum(e_sv)./n_repeat,'--')
hold off
legend('expected error rates','supervised error rates')
ylabel('averaged error(50 experiments)');
xlabel('num. of added unlabeled samples');


%% SVM initialization
% get useful info
Cl = 10000;     % set C
Cu = 0;
[w,xi,b] = SVM_S(X_labeled,X_unlabeled,Cl,Cu);
y_hat = sign(X_unlabeled.data*w+b);
X_unlabel = prdataset(X_unlabeled,y_hat);
%% TSVM
% form the optimization problem
Cl = 10000;     % set C
Cu = 1;
repeat = 1;
err = [];
for i=1:repeat
    while Cu<Cl
        [w_next,xi_next,b_next] = SVM_S(X_labeled,X_unlabel,Cl,Cu);
        y_next = sign(X_unlabeled.data*w_next+b_next);
        fault = sign(y_hat-y_next);
        while sum(fault)~= 0
            disp('not equal to zero')
            predicted_labs = sign(train_u*w+b);
            [v,index] = sign(predicted_labs-original_labs);
        end
        Cu = min(2*Cu,Cl);
    end
    error = sum(abs(y_next-labels_u))/u_num(8);
    err = [err;error];
end
plot(err);



    %% TSVM
    % form the optimization problem
Cl = 10000;     % set C
Cu = 0.5;
repeat = 10;

for i=1:repeat
    
    while Cu<Cl
        [w,xi,b] = SVM_S(X_train,Cl,Cu);
        y_new = sign(train_u*w+b);
        [v,i] = sign(y_hat-y_new);
        while sum(v)~= 0
            cvx_begin
            variables w(dim) xi(n) b        % optim variables
            minimize( 0.5 * (w' * w) + Cl * sum(xi)+Cu*sum(xi))      % objective
            subject to 
            labels_train .* (train_l * w + b) -1 + xi >= 0;   % constraints
            labels_Utrain.* (train_u*w+b) -1 + xi >= 0;
            xi >= 0;
        cvx_end
        predicted_labs = sign(train_u*w+b);
        [v,index] = sign(predicted_labs-original_labs);
        end
    end
    Cu = min(2*Cu,Cl);
end

    
%%
u_num = [0,10,20,40,80,160,320,640];
n_repeat = 5;
e = zeros(n_repeat,length(u_num));
e_sv = zeros(n_repeat,length(u_num));
for r = 1:n_repeat
    index = randperm(size(magic04,1),25);
    train_l = X(index,:);
    a_l = prdataset(train_l,g(index));
    w2 = fisherc(a_l);
    e(r,1) = a_l*w2*testc;
    e_sv(r,1) = a_l*w2*testc;
    for i = 2:length(u_num)
        tst_i = randperm(size(magic04,1),u_num(i));
        tst_lab = g(tst_i);
        tst = prdataset(X(tst_i,:),tst_lab);
        tst_com = [tst; a_l];
        E = tst_com*w2*testc;
        w = fisherc(tst_com);
        e_sv(r,i) = tst_com*w*testc;
        e(r,i)=E;
    end
end
plot(u_num,sum(e)./n_repeat);
hold on
plot(u_num,sum(e_sv)./n_repeat,'--')
hold off
legend('expected error rates','supervised error rates')
ylabel('averaged error(50 experiments)');
xlabel('num. of added unlabeled samples');

% semi-supervised error rates

% supervised error rates