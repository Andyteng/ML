clc
clear all
close all
%% Load the MAGIC Gamma Telescope Data Set
load magic04.mat ; %Load input data file
load labels.mat;
index_g = find([g{:}] == 'g');
index_h = find([g{:}] == 'h');
labels = zeros(1,size(magic04,1));
labels(index_g) = 1;
labels(index_h) = 0;
X = ones(size(magic04,1),10);
for i = 1:10
    X(:,i) = magic04(:,i)./std(magic04(:,i)); 
end
%% 1NN


% unlabeled dataset
    u_num = [0,10,20,40,80,160,320,640];
    index = randperm(size(magic04,1),25);
    train_l = X(index,:);
    train_lab = g(index);
    class1_index = find([train_lab{:}] == 'g');
    class2_index = find([train_lab{:}] == 'h');
    class1_data = train_l(class1_index,:);
    class2_data = train_l(class2_index,:);
    a_l = prdataset(train_l,train_lab);
    w2 = fisherc(a_l);
    e_super = a_l*w2*testc;
    index_u = 1:size(magic04,1);
    index_u(index)=[];
    index_u = randperm(size(index_u,2),u_num(2));
    label_u = g(index_u);
    train_u = X(index_u,:);
    
    %% ML_estimate
    err_num = 0;
    while size(train_u,1) ~= 0
        [mu1_hat, s1_hat] = gaussian_ML_estimate(class1_data');  
        [mu2_hat, s2_hat] = gaussian_ML_estimate(class2_data'); 
        mu_hat = [mu1_hat, mu2_hat];  
        s_hat = (1/2) * (s1_hat + s2_hat); 
        [v, z_euclidean] = euclidean_classifier(mu_hat, train_u');
        [v_min, i_min] = min(v);
        if z_euclidean(i_min) == 1
            [class1_data, train_u,label_u] = update_dataset(i_min,class1_data,train_u,label_u);
            fprintf('sample %i is added to class1\n', i_min);
            if [label_u(i_min)] ~= 'g'
                err_num = err_num+1;
            end
        else 
            [class2_data, train_u,label_u] = update_dataset(i_min,class2_data,train_u,label_u);
            fprintf('sample %i is added to class2\n', i_min);
            if [label_u(i_min)] ~= 'h'
                err_num = err_num+1;
            end
        end
    end
            fprintf('with error number: %i', err_num)
            disp('all unlabeled data were labeled.')
    
    %% TSVM
    % form the optimization problem
Cl = 10000;     % set C
Cu = 0.5;
cvx_begin
    variables w(dim) xi(n) b        % optim variables
    minimize( 0.5 * (w' * w) + Cl * sum(xi)+Cu*sum(xi))      % objective
    subject to 
    labels_train .* (X_train * w + b) -1 + xi >= 0;   % constraints
    labels_Utrain.* (X_Utrain*w+b) -1 + xi >= 0;
    xi >= 0;
cvx_end
%
            
            %%
%     train_diag = diag(train_l*train_l');
%     unlabel_diag = diag(train_u*train_u');
%     
%     dist = repmat(train_diag,[1,10])+repmat(unlabel_diag,[1,25])'-2*train_l*train_u';
    
    
%%
u_num = [0,10,20,40,80,160,320,640];
n_repeat = 50;
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