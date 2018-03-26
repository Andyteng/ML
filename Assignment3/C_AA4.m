function [C] = C_AA4(zt, d,t)

temp_C = 0;

for j = 1:d
    temp_C = temp_C + exp(-L_AA4(zt,j,t));
end

C = temp_C;
