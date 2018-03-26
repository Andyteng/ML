function [C] = C_AA(d,t)

temp_C = 0;

for j = 1:d
    temp_C = temp_C + exp(-L_AA(j,t));
end

C = temp_C;
