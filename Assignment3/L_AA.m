function [L] = L_AA(i,t)

temp_L = 0;
zt = [0 0 1 0; 0.1 0 0 0.9; 0.2 0.1 0 0];

for s = 1:t
    temp_L = temp_L + zt(i,s);
end

L = temp_L;

