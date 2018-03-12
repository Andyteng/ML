function [ v, c, i ] = euclidean_classifier( m, X )  
[~, c] = size(m);  
[~, n] = size(X);  
  
%z = zeros(n, 1);
v = zeros(n,1);
%de = zeros(c, 1);  
% for i = 1:n  
%     for j = 1:c  
%         de(j) = sqrt( (X(:,i)-m(:,j))' * (X(:,i)-m(:,j)));  
%     end  
%     [v(i), z(i)] = min(de);  
% end  

m_diag = diag(m'*m);
unlabel_diag = diag(X'*X);
dist = repmat(m_diag,[1,length(v)])'+repmat(unlabel_diag,[1,c])-2*X'*m;
if size(dist,1) ~= 1
    [v, indices]= min(dist);
    [~, c] = min(v);
    i = indices(c);
else
    [v,c] = min(dist);
    i =1;
end 

