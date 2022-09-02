function [obj,phibeta,gbeta] = acfobj(param,inp)

capital1 = inp.capital1;
labor1 = inp.labor1;
material1 = inp.material1;
energy1 = inp.energy1;
klag = inp.klag;
mlag = inp.mlag;
llag = inp.llag;
elag = inp.elag;
n = inp.n;
nt1 = inp.nt1;

% getting residuals and phi
[etaeps,eta,phibeta,gbeta,~]= residualfnACF(param,inp);

% Instruments
Zetaeps = [ones(nt1,1),capital1, llag, klag, mlag,llag.^2,klag.^2,mlag.^2,capital1.^2];
Zeta = [ones(nt1,1),capital1,llag, klag, mlag,labor1, material1,llag.^2,klag.^2,mlag.^2,capital1.^2,labor1.^2,material1.^2];


Z = [Zeta,Zetaeps];

ip = size(Z,1);

m1t = eta.*Zeta; %obs-by-nZeta
m2t = etaeps.*Zetaeps; %obs-by-nZetaeps
momi = [m1t,m2t];

m1 = mean(m1t,1)'; %(nZeta=3)-by-1
m2 = mean(m2t,1)'; %(nZetaeps=4)-by-1

moment = [m1;m2]; %(nZeta+nZetaeps =7)-by-1

%[fininst,isr]=instref(Z);

%W = (Z'*Z)/ip; % initial weighting matrix
W = cov(momi);

invW = inv(W);

obj = moment'*(W\moment);
end


