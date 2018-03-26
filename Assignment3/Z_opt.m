% This function is used to calculate the min term for expert regret

function [m] = Z_opt(zt,n,d)

ms = zeros(1,d);
for i = 1:d
    temp_m = 0;
    for t = 1:n
        temp_m = temp_m + zt(i,t);
    end
    ms(1,i) = temp_m;
end

[m,~] = min(ms);