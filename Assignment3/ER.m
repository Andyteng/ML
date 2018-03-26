% This function is used to calculate the expect regret
function [R] = ER(pt, zt, n, d)

temp_R = 0;
for t = 1:n
    temp_R = temp_R + mixloss(pt(t,:),zt(:,t));
end

R = temp_R-Z_opt(zt,n,d);
