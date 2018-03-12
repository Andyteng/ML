function [ m_hat, s_hat ] = gaussian_ML_estimate( X )  

[~, N] = size(X);  
m_hat = (1/N) * sum(transpose(X))';  
s_hat = zeros(1);  
for k = 1:N  
    s_hat = s_hat + (X(:, k)-m_hat) * (X(:, k)-m_hat)';  
end  
s_hat = (1/N)*s_hat;  
end  
