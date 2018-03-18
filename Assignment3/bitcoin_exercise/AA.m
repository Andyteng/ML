% Exercise: Aggregating Algorithm (AA)

clear all;
load coin_data;

d = 5;
n = 213;

% compute adversary movez z_t
%%% your code here %%%

% compute strategy p_t (see slides)
%%% your code here %%%

% compute loss of strategy p_t
%%% your code here %%%

% compute losses of experts
%%% your code here %%%

% compute regret
%%% your code here %%%

% compute total gain of investing with strategy p_t
%%% your code here %%%

%% plot of the strategy p and the coin data

% if you store the strategy in the matrix p (size n * d)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(p)
legend(symbols_str)
title('rebalancing strategy AA')
xlabel('date')
ylabel('confidence p_t in the experts')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
