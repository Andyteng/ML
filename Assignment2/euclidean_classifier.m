function [ v, z ] = euclidean_classifier( m, X )  
%{  
    ?????  
        ???????????????  
  
    ?????  
        m????????ML????????????  
        X??????????  
  
    ?????  
        z????????  
%}  
  
[~, c] = size(m);  
[~, n] = size(X);  
  
z = zeros(n, 1);
v = zeros(n,1);
de = zeros(c, 1);  
for i = 1:n  
    for j = 1:c  
        de(j) = sqrt( (X(:,i)-m(:,j))' * (X(:,i)-m(:,j)));  
    end  
    [v(i), z(i)] = min(de);  
end  
  
end 