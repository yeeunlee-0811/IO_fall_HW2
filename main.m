clear

%% Data Loading
% data period: 1979-1996 only include food industry 
data0 = importdata('chile.csv');
data1 = data0.data;

%% drop obs with negative material 
material = data1(:,7);

inm = material<0;
nnm = sum(inm); %13
res_inm = find(inm ==1);

data1(res_inm,:) = [];
data = data1; % 20730-13 = 20717 obs

%% Generating variables
xk = size(data,2);
n = size(data,1);

icode = data(:,1);
pid = data(:,2);
yr0 = data(:,3);
routput = data(:,4);
labor = data(:,5);
energy= data(:,6);
material = data(:,7);
capital = data(:,8);
invest = data(:,9);


y = log(routput);
k = log(capital);
l = log(labor);
m = log(material);
e = log(energy);

% yr
uniqyr0 = unique(yr0); % total 18 years: 1979-1996
nyr = size(uniqyr0,1); 
uniqyr = uniqyr0 - ones(nyr,1)*1978;

yr = zeros(n,1);

for i = 1:nyr
    iyr = uniqyr0(i,1) == yr0;
    yr(iyr,1) = uniqyr(i,1);
end

% plant

p0 = unique(pid);% total 2673 plants
np = size(p0,1);
%%
inp.np = np;
inp.p0 = p0;
inp.yr = yr;
inp.uniqyr = uniqyr;
inp.nyr = nyr;
inp.uniqyr0 = uniqyr0;
inp.invest = invest;
inp.k = k;
inp.m = m;
inp.energy = energy;
inp.l = l;
inp.y = y;
inp.n = n;
inp.xk = xk;
inp.icode = icode;
inp.pid = pid;
inp.yr0 = yr0;



%% Q1 - a: OLS 

% Simple OLS

Xols = [k, l];

ols_simple = fitlm(Xols,y);

%% Q1 - b: Fixed effects - mean difference

KLmean = zeros(n,1);
for i = 1:np
    ip = p0(i,1) == pid;
    KLmean(ip,1) = mean(k(ip));
    KLmean(ip,2) = mean(l(ip));
end

Xols_mean = KLmean;

Xmeandiff = Xols - Xols_mean;

ols_meandiff = fitlm(Xmeandiff, y,'Intercept',false);
%% Generating lags and contemporaneous data

% iinit: logical vector of first obs of each firm
iinit = geniinit(inp); 

%ifin: logical vector of the last obs of each firm
iinitd = double(iinit);
iinitd(1,:) = [];
iinitd(end+1,1) = 1;
ifin = logical(iinitd); 

% lagged data
llag = l(~ifin);
ylag = y(~ifin);
mlag = m(~ifin);
klag = k(~ifin);
elag = e(~ifin);

% contemporaneous
labor1 = l(~iinit);
routput1 = y(~iinit);
capital1 = k(~iinit);
material1 = m(~iinit);
energy1 = e(~iinit);
nt1 = size(material1,1); % the number of contemporaneous obs



inp.ifin = ifin;
inp.iinit = iinit;
inp.llag = llag;
inp.klag = klag;
inp.ylag = ylag;
inp.mlag = mlag;
inp.elag = elag;
inp.labor1 = labor1;
inp.routput1 = routput1;
inp.capital1 = capital1;
inp.material1 = material1;
inp.energy1 = energy1;
inp.nt1 = nt1;

%% Q1 - b: Fixed effects - First difference
Xols_lag = [klag,llag];

Xols_diff = Xols(~iinit,:) - Xols_lag;

y_diff = routput1 - ylag;


ols_firstdiff = fitlm(Xols_diff, y_diff,'Intercept',false);

%% Q1 - b: Fixed effects - Dummy variables

pdummy = zeros(n,np);

for i = 1:np
    ip = pid == p0(i);
    
    pdummy(ip,i) = 1;
end

pdummy(:,1) = []; % drop a base plant dummy

Xols_fe = [k, l, pdummy];

ols_fe = fitlm(Xols_fe,y);

ols_fe_res = ols_fe.Coefficients;

save('ols_fe_param.mat','ols_fe_res')

%% Q1-c: LP with Wooldridge moments 

k_lp = 3; % the number of parameters we estimate: beta0, beta_l, beta_k
inp.k_lp2 = k_lp;

theta0 = rand(k_lp,1);
obj = @(theta)lpobj(theta, inp);

options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxFunEvals',10e+10,'TolFun',1.0e-10);

[lptheta0,logval0] = fminsearch(obj,theta0,options); %first stage GMM
[lpthetafin,logvalfin] = fminsearch(obj,lptheta0,options); %second stage GMM

%% Q1-d: ACF with Wooldridge moments 

k_ACF = 3; % the number of parameters we estimate: beta0, betal, betak
inp.k_ACF = k_ACF;

ntrial = 10;

theta0_ACF = rand(k_ACF,1);
objACF = @(theta)acfobj(theta, inp);

options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxFunEvals',10e+10,'TolFun',1.0e-10);

[acftheta0,acflogval0] = fminsearch(objACF,theta0_ACF,options); %first stage GMM
[acfthetafin,acflogvalfin] = fminsearch(objACF,acftheta0,options); %second stage GMM

%% Q2-Hicksian shocks using OLS result

nsimple = size(Xols,1);

loghshockols = y - Xols*ols_simple.Coefficients.Estimate(2:end,1); 

histogram(loghshockols,100)
save('loghshockols.mat','loghshockols');

varshock_ols = var(loghshockols);

%% Q2-Fixed effects

loghshockfe = routput1 - Xols(~iinit,:)*ols_firstdiff.Coefficients.Estimate; 

histogram(loghshockfe,100)
save('loghshockfe.mat','loghshockfe');

varshock_fe = var(loghshockfe);
%% Q2-Hicksian shocks using ACF result

[~,etaacf,~,~,omegaacf]= residualfnACF(acfthetafin,inp);

beta0acf = acfthetafin(1,1);
loghshockacf = beta0acf*ones(nt1,1)+omegaacf+etaacf;
hshockacf = exp(loghshockacf);

histogram(loghshockacf,100)
save('loghshockacf.mat','loghshockacf');

varshock_acf = var(loghshockacf);
%% Q2-Hicksian shocks using LP result

[etalp,~,~,~,omegalp] = residualfn(lpthetafin,inp);

beta0lp =  lpthetafin(1,1);
loghshocklp = beta0lp*ones(nt1,1)+omegalp+etalp;
hshocklp = exp(loghshocklp);

save('loghshocklp.mat','loghshocklp');
histogram(loghshocklp,100)

varshock_lp = var(loghshocklp);

%% Q3-Block bootstrap - block by year (Total 18 years)

% YKLM matrix: first 4 columns -> contemporaenous 
% & 5-8 -> lagged
YKLM = [routput1,capital1,labor1,material1,ylag,klag,llag,mlag]; 

yrk = yr(~iinit);

% YKLMblck matrix: reorder data by block(same yr)
YKLMblck = [];

obsbyyr = zeros(nyr,1); % N of obs by yr
for i = 1:nyr
    iyr = i == yrk;
    obsbyyr(i,1) = sum(iyr);
    YKLMblck = [YKLMblck ; YKLM(iyr,:)];
end

%% Generating Bootstrap Sample: 50 sampels with size 9966

ns= 60;

obsbyyr0 = [0;obsbyyr]; %19-by-1
obsbyyr1 = cumsum(obsbyyr0);

sboot = zeros(9966,8,ns);
s0 =[];
for z = 1:ns
        for i = 1:nyr
            dta = YKLMblck(obsbyyr1(i,1)+1:obsbyyr1(i+1,1),:);
            s0 = [s0;datasample(dta,round(obsbyyr(i,1)/n*10000))];
        end
    sboot(:,:,z) = s0;
    s0 = [];
end

%% OLS

betaboots_OLS = zeros(ns,3);

for z= 1:ns
    dta = sboot(:,:,z);
    
    % resetting input data
    inp.routput1 = dta(:,1);
    inp.capital1 = dta(:,2);
    inp.labor1 = dta(:,3);
    inp.material1 = dta(:,4);
    inp.nt1 = size(dta,1);
    
    inp.ylag = dta(:,5);
    inp.klag = dta(:,6);
    inp.llag = dta(:,7);
    inp.mlag = dta(:,8);
    
    Xols_boots = [dta(:,2), dta(:,3)];
    
    ols_simple_boots = fitlm(Xols_boots,dta(:,1));
    

    betaboots_OLS(z,:) = ols_simple_boots.Coefficients.Estimate';
    
end


%% FE-first difference 

betaboots_fe = zeros(ns,2);

for z= 1:ns
    dta = sboot(:,:,z);
    
    % resetting input data
    inp.routput1 = dta(:,1);
    inp.capital1 = dta(:,2);
    inp.labor1 = dta(:,3);
    inp.material1 = dta(:,4);
    inp.nt1 = size(dta,1);
    
    inp.ylag = dta(:,5);
    inp.klag = dta(:,6);
    inp.llag = dta(:,7);
    inp.mlag = dta(:,8);
    
    Xols_boots = [dta(:,2),dta(:,3)];
    Xols_lag_boots = [dta(:,6),dta(:,7)];
    
    Xols_diff_boots = Xols_boots - Xols_lag_boots;
    
    y_diff = dta(:,1) - dta(:,5);
    
    ols_firstdiff_boots = fitlm(Xols_diff_boots, y_diff,'Intercept',false);
    
    betaboots_fe(z,:) = ols_firstdiff_boots.Coefficients.Estimate';
    
end
%% ACF

betaboots_acf = zeros(ns,3);

for z= 1:ns
    dta = sboot(:,:,z);
    
    % resetting input data
    inp.routput1 = dta(:,1);
    inp.capital1 = dta(:,2);
    inp.labor1 = dta(:,3);
    inp.material1 = dta(:,4);
    inp.nt1 = size(dta,1);
    
    inp.ylag = dta(:,5);
    inp.klag = dta(:,6);
    inp.llag = dta(:,7);
    inp.mlag = dta(:,8);
    
    % ACF process
    objACF = @(theta)acfobj(theta, inp);
    
    options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxFunEvals',10e+10,'TolFun',1.0e-10);
    
    [acftheta_boots,acflogval0] = fminsearch(objACF,acfthetafin,options);
    
    % getting params
    betaboots_acf(z,:) = acftheta_boots';
    
end

%% LP

betaboots_lp = zeros(ns,3);

for z= 1:ns
    dta = sboot(:,:,z);
    
    % resetting input data
    inp.routput1 = dta(:,1);
    inp.capital1 = dta(:,2);
    inp.labor1 = dta(:,3);
    inp.material1 = dta(:,4);
    inp.nt1 = size(dta,1);
    
    inp.ylag = dta(:,5);
    inp.klag = dta(:,6);
    inp.llag = dta(:,7);
    inp.mlag = dta(:,8);
    
    % LP process
    obj = @(theta)lpobj(theta, inp);
    
    options = optimset('Display','iter','PlotFcns',@optimplotfval,'MaxFunEvals',10e+10,'TolFun',1.0e-10);
    
    [lptheta_boots,~] = fminsearch(obj,lpthetafin,options); 

    
    % getting params
    betaboots_lp(z,:) = lptheta_boots';
    
end

%% Betaboots ACF

save('betaboots_acf.mat','betaboots_acf');

covbetaboots_acf = cov(betaboots_acf);
sebetaboots_acf = sqrt(diag(covbetaboots_acf));
tstat_acf = acfthetafin./sebetaboots_acf;

%% Betaboots lp

save('betaboots_lp.mat','betaboots_lp');
covbetaboots_lp = cov(betaboots_lp);
sebetaboots_lp = sqrt(diag(covbetaboots_lp));
tstat_lp = lpthetafin./sebetaboots_lp;


%% Betaboots OLS

save('betaboots_OLS.mat','betaboots_OLS');
covbetaboots_OLS = cov(betaboots_OLS);
sebetaboots_OLS = sqrt(diag(covbetaboots_OLS));
tstat_OLS = ols_simple.Coefficients.Estimate./sebetaboots_OLS;

%% Betaboots fe

save('betaboots_fe.mat','betaboots_fe');
covbetaboots_fe = cov(betaboots_fe);
sebetaboots_fe = sqrt(diag(covbetaboots_fe));
tstat_fe = ols_firstdiff.Coefficients.Estimate./sebetaboots_fe;