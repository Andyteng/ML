function [L] = L_AA4(zt,i,t)

temp_L = 0;

for s = 1:t
    temp_L = temp_L + zt(s,i);
end

L = temp_L;

