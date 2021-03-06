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

% TSVM
% form the optimization problem
repeat = 10;
e_history_svm = zeros(9,repeat);
LL_svm = zeros(9,repeat);
l = 25;

[X_labeled, rest] = gendata_split(l,X_dataset);

for i=1:repeat
    for j = 1:length(u_num)
        % no unlabled dataset
        if u_num(j) == 0
            [X_labeled, rest] = gendata_split(25,X_dataset);
            w = svc(X_labeled,'p',1);
            y_hat = X_dataset*w*labeld;
            e_history_svm(j,i) = sum(X_dataset.labels~=y_hat)/size(X_dataset.data,1);
            
            X_pre = prdataset(X_dataset.data,y_hat);
            ll_svm = ll(X_pre);
            LL_svm(j,i) = ll_svm;
        else
            [X_unlabeled,X_test] = gendata_split(u_num(j),rest);
            [w,xi,b] = SVM_S(X_labeled,X_unlabeled,Cl,Cu);
            y_hat =   sign(X_unlabeled.data*w+b);
            X_uunlabeled = prdataset(X_unlabeled.data,y_hat);
            while Cu<Cl
                [w_n,xi_n,b_n] = SVM_S(X_labeled,X_uunlabeled,Cl,Cu);
                xi_u = xi_n(l+1:end);
                for m = 1:size(X_unlabeled.data,1)
                    for n = 1:size(X_unlabeled.data,1)
                        if y_hat(m)*y_hat(n)<0 && xi_u(m>0) && xi_u(n>0) && (xi_u(m)+xi_u(n))>2
                            y_hat(m) = -y_hat(m);
                            y_hat(n) = -y_hat(n);
                        end
                    end
                end
                X_uunlabeled = prdataset(X_unlabeled.data,y_hat);
                [w,xi,b] = SVM_S(X_labeled,X_unlabeled,Cl,Cu);
                Cu = min(2*Cu,Cl);
                fprintf("The current value of Cu is: %i" + Cu)
            end
            y_hat = sign( X_dataset.data * w + b);
            error = sum(X_dataset.labels~=y_hat)/size(X_dataset.data,1);
            e_history_svm(j,i) = error;
            
            X_pre = prdataset(X_dataset.data,y_hat);
            ll_svm = ll(X_pre);
            LL_svm(j,i) = ll_svm;
        end
    end
end

%% basic LDA testing
% [X_labeled, rest] = gendata_split(25,X_dataset); 
% X = X_labeled;
% [mean0, mean1, sigma,prior0] = llda(X,25, 500);
% [e, ll] = llda_predict(rest,mean0,mean1,sigma,prior0);

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
%% EM 
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
        [mean0, mean1, sigma,prior0] = llda(X, 25, 300);
        [e,ll]= llda_predict(X_test,mean0,mean1,sigma,prior0);
        e_history(i,j) = e; 
        LL(i,j) = ll;
    end
end
plot(u_num,mean(e_history,2))
hold on
plot(u_num,mean(e_history_svm,2))
hold off
legend('GM','TSVM');
ylabel('averaged error(50 experiments)');
xlabel('num. of added unlabeled samples');
figure;
plot(u_num,mean(LL,2))
legend('TSVM')
ylabel('averaged loglikelihood(50 experiments)');
xlabel('num. of added unlabeled samples');

%% EM

% u_num = [0,10,20,40,80,160,320,640];
% n_repeat = 5;
% e = zeros(n_repeat,length(u_num)-1);
% e_sv = zeros(n_repeat,length(u_num));
% for r = 1:n_repeat
%     err = [];
%     % labeled dataset
%     [X_labeled, rest] = gendata(25,X_dataset); 
%     label_l = X_labeled.labels;
%     train_l = X_labeled.data;
% 
%     class1_index = find(label_l == 1);
%     class2_index = find(label_l == 0);
%     class1 = X_labeled(class1_index,:);
%     class2 = X_labeled(class2_index,:);
% 
%     %supervised error rate
%     err_supervised = X_dataset*ldc(X_labeled)*testc;
%     u_num = [0,10,20,40,80,160,320,640];
%     for i = 2:length(u_num)
%         % unlabeled dataset
%         [X_unlabeled,rrest] = gendata(u_num(i),rest);
%         label_u = X_unlabeled.labels;
%         train_u = X_unlabeled.data;
%         err_num = 0;
%         count = 0;
%         y_hat = zeros(size(label_u,1),1);
%         %fprintf("%i unlabeled samples are added into the training dataset\n", u_num(i))
%         while size(X_unlabeled.data,1) ~= 0
%             %fprintf("%i samples were labeled \n", count);
%             [mu1_hat, s1_hat] = gaussian_ML_estimate(class1.data');  
%             [mu2_hat, s2_hat] = gaussian_ML_estimate(class2.data'); 
%             mu_hat = [mu1_hat, mu2_hat];  
%             s_hat = (1/2) * (s1_hat + s2_hat); 
%             [v_min, c, i_min] = euclidean_classifier(mu_hat, X_unlabeled.data');
%             if c == 1
%                 [class1, X_unlabeled] = update_dataset(i_min,class1,X_unlabeled);
%                 y_hat(i_min+count) = 1;
%                 count= count+1;
%                 %fprintf('sample %i is added to class1\n', i_min);
%                 if label_u(i_min) ~= 1
%                     err_num = err_num+1;
%                 end
%             else 
%                 [class2, X_unlabeled] = update_dataset(i_min,class2,X_unlabeled);
%                 y_hat(i_min) = 1;
%                 count= count+1;
%                 %fprintf('sample %i is added to class2\n', i_min);
%                 if label_u(i_min) ~= 0
%                     err_num = err_num+1;
%                 end
%             end
%         end
%         fprintf('with error number: %i \n', err_num)
%         disp('all unlabeled data were labeled.')
%         X_uunlabeled = prdataset(train_u,y_hat);
%         X_llabeled = [X_labeled;X_uunlabeled];
%         error = X_dataset*fisherc(X_llabeled)*testc;
%         err = [err; error];
%     end
%     e(r,:) = err;
% end
% plot(u_num(2:end),sum(e)./n_repeat);
% hold on
% plot(u_num(2:end),sum(e_sv)./n_repeat,'--')
% hold off
% legend('expected error rates','supervised error rates')
% ylabel('averaged error(50 experiments)');
% xlabel('num. of added unlabeled samples');
