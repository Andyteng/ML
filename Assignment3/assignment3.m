 clc; clear all; close all;

%% Q1a

% strategy A
d = 3; % number of experts
end_t = 4;
e = eye(d);
temp_lm = zeros(1,d);
pt_A = zeros(end_t,d);
p1 = [1/3 1/3 1/3];
pt_A(1,:) = p1;
zt = [0 0 1 0; 0.1 0 0 0.9; 0.2 0.1 0 0];

for t = 2:end_t
    for i = 1:d
        temp_lm(1,i) = cumloss(e,zt,i,t);
    end
    [~,bt] = min(temp_lm);
    pt_A(t,:) = e(bt,:);
end 

%% Strategy B: AA cumulative loss

d = 3; % number of experts
end_t = 4;
pt_B = zeros(end_t,d);
p1 = [1/3 1/3 1/3];
pt_B(1,:) = p1;

for t = 2:end_t
    for i = 1:d
        pt_B(t,i) = exp(-L_AA(i,t-1))/C_AA(d,t-1);
    end
end

%% Q1b
d = 3;
end_t = 4;
ml_A = 0;
ml_B = 0;
% The mix loss for Strategy A
for t = 1:end_t
    ml_A = ml_A + mixloss(pt_A(t,:),zt(:,t));
end
% The mix loss for Strategy B
for t = 1:end_t
    ml_B = ml_B + mixloss(pt_B(t,:),zt(:,t));
end

% Expect regret
temp_A = 0;
temp_B = 0;

for n = 1:end_t
    temp_A = temp_A + ER(pt_A,zt,n,d);
    temp_B = temp_B + ER(pt_B,zt,n,d);
end
R_A = temp_A;
R_B = temp_B;

