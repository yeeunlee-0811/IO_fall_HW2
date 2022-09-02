function [etaeps,eta, phibeta,gbeta,omega]= residualfnACF(param,inp)

beta0 = param(1,1);
betak = param(2,1);
betal = param(3,1);

capital1 = inp.capital1;
material1 = inp.material1;
labor1 = inp.labor1;
routput1 = inp.routput1;
klag = inp.klag;
mlag = inp.mlag;
llag = inp.llag;
ylag = inp.ylag;
nt1 = inp.nt1;

%% 1st stage

phieta = routput1 ;

Xphi = [ones(nt1,1),capital1, labor1, material1,capital1.^2, labor1.^2, material1.^2, capital1.*labor1, capital1.*material1, material1.*labor1, capital1.*labor1.*material1];
k_phi = size(Xphi,2);

[phires] = fitlm(Xphi,phieta,'intercept',false);

phibeta = phires.Coefficients.Estimate;

phi1 = Xphi*phibeta;

Xphilag = [ones(nt1,1),klag, llag, mlag, klag.^2, llag.^2, mlag.^2, klag.*llag, klag.*mlag, mlag.*llag, mlag.*klag.*llag];

philag = Xphilag*phibeta;

%% estimage g function

omegalag = philag - beta0*ones(nt1,1) - betal*llag - betak*klag ;

Xomega = [ones(nt1,1),omegalag,omegalag.^2];
k_omega = size(Xomega,2);

gxi = phi1 - beta0*ones(nt1,1) - betak*capital1 - betal*labor1;
gres = fitlm(Xomega,gxi,'intercept',false);

gbeta = gres.Coefficients.Estimate;

gfit = Xomega*gbeta;
%gfit = fit(omegalag,gxi,'smoothingspline');

%% Getting etaeps and eta and omega

omega = phi1 - beta0*ones(nt1,1) - betal*labor1 - betak*capital1;

etaeps = routput1 - beta0*ones(nt1,1) - betal*labor1 - betak*capital1 - gfit; 
eta = routput1 - phi1;

end


