%% Initialization
clear all; close all; clc
%% Assignment 1
% Question 1a
% parameter
x1 = -1;
x2 = 1;
r_p = -4:.02:4;
lbd = [0 1 2 100];
m = zeros(1,4);
m_gra = zeros(1,4);


%%
% calculation
for i = 1:1:4
L = (1/2)*((x1-r_p).^2+(x2-r_p).^2)+lbd(i)*abs(1-r_p);
FX(i,:) = gradient(L);
[m_gra(i),ind_gra] = min(abs(FX(i,:)));
co_gra_x(i) = r_p(ind_gra);
co_gra_y(i) = L(ind_gra);
[m(i),ind] = min(L);
co_x(i) = r_p(ind);
plot(r_p,L);
hold on;
end

legend ('lbd=0','lbd=1','lbd=2', 'lbd=3');
xlabel ('r_+')
ylabel ('L(r_-,r_+)');
title ('Loss function as a function of r_+');
%% ======================= Question 3: Experimenting =======================
data = load('optdigitsubset.txt');
num = length(data);
X = data((1:num),:);
class0 = X((1:554),:);
class1 = X((555:end),:);

% initial estimate
r_p = ones(1,64);
h = -0.001;
dev = (2/571)*sum((repmat(r_p,[571,1])-class1),1)+ones(1,64)*10;
while sum(abs(dev)) > 0.001
    disp(sum(abs(dev)))
    r_np = r_p +h.*dev;
    dev = (2/571)*sum((repmat(r_np,[571,1])-class1),1)+ones(1,64)*10;
    r_p = r_np;
end

%% lambda is large
data = load('optdigitsubset.txt');
num = length(data);
X = data((1:num),:);
r = ones(1,64);
h = -0.001;
dev = (2/1125)*sum((repmat(r,[1125,1])-X),1);
while sum(abs(dev)) > 0.001
    disp(sum(abs(dev)))
    r_p = r +h.*dev;
    dev = (2/1125)*sum((repmat(r_p,[1125,1])-X),1);
    r = r_p;
end


%% Question 3b
% mp
img = reshape(r_p,[8,8]);
img=transpose(img);
img = mat2gray(img);
figure
imshow(img,'InitialMagnification','fit'); %fit the screen
% mn
img = reshape(mn,[8,8]);
img=transpose(img);
img = mat2gray(img);
figure
imshow(img,'InitialMagnification','fit'); %fit the screen

%% Question 3c
t_0 = class0(randi([1 554]),:);
t_1 = class1(randi([1 557]),:);
dist_0 = (1/554)*sum((X-repmat(t_0,[1125,1])).^2,2);
dist_1 = (1/571)*sum((X-repmat(t_1,[1125,1])).^2,2);
y = sign(dist_0-dist_1);
y_0 = y(1:554);
y_1 = y(555:end);
true_err = (sum(y_0 == 1) + sum(y_1 == -1))/1123;

r_p = ones(1,64);
r_n = ones(1,64);
lambda = 0;

cvx_begin
variable r_p,r_n
minimize (sum(sum((1/554)*(class0-repmat(r_p,[554,1])).^2,2))+sum(sum((1/571)*(class1-repmat(r_n,[571,1])).^2,2))+lambda * norm(r_n-r_p,1))
cvx_end

%%
load('optdigitsubset.txt');
[n,m]=size(optdigitsubset);
subset1 = (optdigitsubset(1:554,:))';
subset2 = (optdigitsubset(555:1125,:))';
K=100;
err1=zeros(K,1);
err2=zeros(K,1);
for k=1:K

train1 = subset1(:,ceil(rand * 554));
train2 = subset2(:,ceil(rand * 571));

lamda = 10;
cvx_begin
    variable A(m);
    variable B(m);
    minimize( ((sum(sum_square(train1 - A))+sum((sum_square(train2 - B))) + lamda * norm(A - B, 1))));
cvx_end

r1 = A';
r2 = B';
if (norm(r1-train1')>norm(r2-train1'))
    err1(k)=1+err1(k);
end
if (norm(r2-train2')>norm(r1-train2'))
    err2(k)=1+err2(k);
end
%{
for i=1:554
if(norm(r1-subset1(:,i)',2)>norm(r2-subset1(:,i)',2))
    err1(k)=1+err1(k);
end
end
for i=1:571
if(norm(r2-subset2(:,i)',2)>norm(r1-subset2(:,i)',2))
    err2(k)=1+err2(k);
end
end
%}
k=k+1;
end
%true_err=0.5*mean(err1)/(554)+0.5*mean(err2)/(571)
app_err = 0.5*mean(err1)+0.5*mean(err2)
