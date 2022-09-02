function [obj,phibeta,gbeta] = lpobj(param, inp)

capital1 = inp.capital1;
labor1 = inp.labor1;
material1 = inp.material1;
klag= inp.klag;
mlag = inp.mlag;
llag = inp.llag;
nt1 = inp.nt1;

% getting residuals
[eta,etaeps,phibeta, gbeta] = residualfn(param,inp);



Zeta = [ones(nt1,1), capital1, labor1, material1];
Zetaeps = [ones(nt1,1),capital1, klag, mlag, llag];

Z = [Zeta,Zetaeps];

ip = size(Z,1);

m1t = eta.*Zeta; %obs-by-nZeta
m2t = etaeps.*Zetaeps; %obs-by-nZetaeps
momi = [m1t,m2t];

m1 = mean(m1t,1)'; %(nZeta=3)-by-1
m2 = mean(m2t,1)'; %(nZetaeps=4)-by-1

moment = [m1;m2]; %(nZeta+nZetaeps =7)-by-1

%W = (Z'*Z)/ip; % initial weighting matrix
W = cov(momi);

invW = inv(W);

obj = moment'*(W\moment);
end














