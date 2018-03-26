% This function is used to calculate the expect
function [R] = ER4(pt, zt, n, d)

temp_R = 0;
for t = 1:n
    temp_R = temp_R + mix_loss(pt(t,:),zt(t,:));
end

R = temp_R-Z_opt(zt,n,d);
