function [L] = cumloss(e,zt,i,t)

ei = e(i,:);
temp_L = 0;

for s = 1:t-1
    temp_L = temp_L + mixloss(ei,zt(:,s));
end

L = temp_L;