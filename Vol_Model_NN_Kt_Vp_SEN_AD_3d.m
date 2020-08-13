%% load data
close all; 
clc; clear ;

addpath(genpath('minFunc_2012'));
addpath(genpath('ComputingFunction'));
addpath(genpath('Training_data'));
addpath(genpath('recon_NN'));
name = 'MODL_patB_20';
data_name  = strcat('recon_',name,'.h5') ;

%%


imgF0 = h5read(data_name,'/field2');
imgF0 = double(imgF0.r + 1i*imgF0.i) ;

%%
 imgF0 = reshape(imgF0,320,320,[],32);
 
% % imgF0 = imgF0(:,:,61,:);
 imgAll = zeros(size(imgF0)) ;
% % imgF = normalize(imgF0);
for i =1:size(imgF0,3)
    a = imgF0(:,:,i,:) ;
    a = a(:);
    a = (a - min(a))./(max(a) - min(a));
    imgAll(:,:,i,:) = reshape(a,size(imgF0(:,:,i,:)));
% imgAll = imgF0/100;
end
k = fft2(imgAll);
%%
opt.size=size(k);  
[kx,ky,kz,nt,ncoil]=size(k);
sMaps = ones([kx,ky,kz]) ;
% sMaps=sMaps(:,:,ns,:,:);
sMaps=reshape(sMaps,[kx ky kz 1 ncoil]);

if ~exist('R1','var')  % use simulated uniform M0 and R1 if none exists
    M0=5*ones(kx,ky,kz,'single'); %use simulated M0, R1
    R1=0.6*ones(kx,ky,kz,'single');
end

imgF=sum(conj(repmat(sMaps,[1 1 1 nt 1])).*ifft2(k),5); % get fully-sampled

clear imgF0
clear imgAll
%% set parameters
opt.wname='db4'; % wavelet parameters
opt.worder={[1 2],[1 2],[1 2]};
opt.R1=R1;
opt.M0=M0;
opt.Sb=repmat(imgF(:,:,:,1),[1 1 1 nt]);  %baseline image
opt.alpha=pi*10/180; %flip angle
opt.TR=0.006;  %TR

delay=3; % delay frames for contrast injection
tpres=20/60; % temporal resolution, unit in seconds!
opt.time=[zeros(1,delay),[1:(nt-delay)]*tpres];
opt.plot=1;  % to plot intermediate images during recon

%% calculate fully-sampled Ktrans and Vp
CONCF = sig2conc2((imgF),R1,M0,opt.alpha,opt.TR);
opt.AIF=SAIF_p(opt.time); % get population-averaged AIF

[Kt_nn,Vp_nn]= conc2Ktrans(CONCF,opt.time,opt.AIF);
%%

% cd('vol')
% savename = strcat(name,'.mat');
% save(savename,'Kt_nn') ;
% cd('..')
