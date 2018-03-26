function [lm] = mixloss(pt,zt)

lm = -log(pt*exp(-zt));