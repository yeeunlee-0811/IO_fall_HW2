function [eta,etaeps,phibeta,gbeta,omega] = residualfn(param,inp)

capital1 = inp.capital1;
material1 = inp.material1;
labor1 = inp.labor1;
routput1 = inp.routput1;

klag = inp.klag;
mlag = inp.mlag;
nt1 = inp.nt1;

beta0 = param(1,1);
betak = param(2,1);
betal = param(3,1);

%% first stage
phieta = routput1 - labor1*betal;

Xphi = [ones(nt1,1),capital1, material1,capital1.^2, material1.^2, capital1.*material1];
%capital.^2, material.^2, capital.*material
k_phi = size(Xphi,2);

[phires] = fitlm(Xphi,phieta,'intercept',false);

phibeta = phires.Coefficients.Estimate;

phi1 = Xphi*phibeta;

Xphilag = [ones(nt1,1),klag, mlag, klag.^2, mlag.^2, klag.*mlag];
philag = Xphilag*phibeta;

%% estimage g function

omegalag = philag - beta0*ones(nt1,1) - betak*klag;

Xomega = [ones(nt1,1),omegalag, omegalag.^2, omegalag.^3];
k_omega = size(Xomega,2);

gxi = phi1 - beta0*ones(nt1,1) - betak*capital1;

gres = fitlm(Xomega,gxi,'intercept',false);

gbeta = gres.Coefficients.Estimate;

gfit = Xomega*gbeta;

%% getting eta+epsilon and eta

% using data that omegalag is observed: drop the first observation of each
% plant
omega = phi1 - beta0*ones(nt1,1) - betal*labor1 - betak*capital1;

etaeps = routput1 - beta0*ones(nt1,1) - betal*labor1 - betak*capital1 - gfit; 
eta = routput1 - betal*labor1 - phi1;


end




