close all
clear all

dx = 0.025; dy = 0.025; dz = 0.025;
nx = 321; ny = 241; nz = 166;
n  = [nz,nx,ny]; N = prod(n); 
h  = [dz*1000 dx*1000 dy*1000];

str =['vn_1_true_3D'];
filename=['' str '.bin'];
fid=fopen(filename,'rb');
v_true=fread(fid,[nz*ny*nx,1],'float');
fclose(fid);

v_true = v_true/1000;

vv = reshape(v_true,n); v = vv(1:2:122,21:4:264,1:3:183);

sigma = 1;
v = imgaussfilt3(v,sigma);

clim = [2.500 3.500];
figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(v, dz,dx,dy, clim, [30*dx], [30*dy], [35*dz], [], ...
    'attributeName', 'V (km/s)', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);


str =['eta_1_true_3D'];
filename=['' str '.bin'];
fid=fopen(filename,'rb');
eta_true=fread(fid,[nz*ny*nx,1],'float');
fclose(fid);

vv = reshape(eta_true,n); eta = vv(1:2:122,21:4:264,1:3:183);
eta = imgaussfilt3(eta,sigma);

clim = [0.000 0.100];
figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(eta, dz,dx,dy, clim, [30*dx], [30*dy], [35*dz], [], ...
    'attributeName', 'Amplitude', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

str =['del_3_true_3D'];
filename=['' str '.bin'];
fid=fopen(filename,'rb');
delta_true=fread(fid,[nz*ny*nx,1],'float');
fclose(fid);

vv = reshape(delta_true,n); delta = vv(1:2:122,21:4:264,1:3:183);
delta = imgaussfilt3(delta,sigma);

clim = [0.000 0.0500];
figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(delta, dz,dx,dy, clim, [30*dx], [30*dy], [35*dz], [], ...
    'attributeName', 'Amplitude', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

n = size(v);
nz = n(1); ny = n(3); nx = n(2);

z  = [0:n(1)-1]'*dz;
x  = [0:n(2)-1]*dx;
y  = [0:n(3)-1]*dy;

[zz,xx,yy] = ndgrid(z,x,y);

x_star = xx(:);
z_star = zz(:);
y_star = yy(:);

%%%Training data
vv0 = 3.2;
x1 = (0:301-1)*0.005;
z1 = (0:301-1)'*0.005;
y1 = (0:301-1)*0.005;
[zz1,xx1,yy1] = ndgrid(z1,x1,y1);


src_x = (nx-1)/2;
src_y = (ny-1)/2;
src_z = (nz-1)/2;

xs = (src_x-1)*dx;
ys = (src_y-1)*dy;
zs = (src_z-1)*dz;

ifre = 10;

omega = 1*2*pi*ifre;
r = @(zz,xx,yy)(zz.^2+xx.^2+yy.^2).^0.5;
% Wavenumber

K = (omega./vv0);
% For 3D case
G3D = @(zz,xx,yy)exp(1i*K.*r(zz,xx,yy))./r(zz,xx,yy);
G_3d = G3D(zz1 - zs, xx1 - xs, yy1 - ys)/7.75;
G_3d = -G_3d;

clim = [-0.2 0.2];


figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(real(G_3d), 0.005,0.005,0.005, clim, [1*0.005], [1*0.005], [0*dz], [], 'attributeName', 'Amplitude');


figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(real(G_3d), 0.005,0.005,0.005, clim, [200*0.005], [200*0.005], [100*0.005], [], ...
    'attributeName', 'Amplitude', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

[G_3d_fx,G_3d_fy,G_3d_fz]=gradient(G_3d);

[G_3d_fxx,G_3d_fxy,G_3d_fzx]=gradient(G_3d_fx);
[G_3d_fyx,G_3d_fyy,G_3d_fzy]=gradient(G_3d_fy);
[G_3d_fzx,G_3d_fzy,G_3d_fzz]=gradient(G_3d_fz);

dzzG_3d = G_3d_fzz/0.005/0.005;
dxxG_3d = G_3d_fxx/0.005/0.005;
dyyG_3d = G_3d_fyy/0.005/0.005;


[Xq,Yq,Zq] = meshgrid(x1,y1,z1);

v_in = interp3(x,y,z,v,Xq,Yq,Zq);
eta_in = interp3(x,y,z,eta,Xq,Yq,Zq);
delta_in = interp3(x,y,z,delta,Xq,Yq,Zq);

clim = [2.500 3.500];
figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(v_in, dz,dx,dy, clim, [30*dx], [30*dy], [35*dz], [], ...
    'attributeName', 'V (km/s)', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);
% 
% clim = [-0.500 0.500];
% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(real(G_3d), dz,dx,dy, clim, [30*dx], [30*dy], [35*dz], [], ...
%     'attributeName', 'V (km/s)', ...
%     'filtCoef', 0.4, ...
%     'startTime', 0.1);

N_train = 50000;

x_train = 1.5*rand(N_train,1) ;
z_train = 1.5*rand(N_train,1) ;
y_train = 1.5*rand(N_train,1) ;

xx_in = round(x_train/0.005)+1;
zz_in = round(z_train/0.005)+1;
yy_in = round(y_train/0.005)+1;

v_train = zeros(N_train,1);
eta_train = zeros(N_train,1);
delta_train = zeros(N_train,1);
U0_imag_train = zeros(N_train,1);
U0_real_train = zeros(N_train,1);
dxxU0_imag_train = zeros(N_train,1);
dxxU0_real_train = zeros(N_train,1);
dyyU0_real_train = zeros(N_train,1);
dyyU0_imag_train = zeros(N_train,1);
dzzU0_real_train = zeros(N_train,1);
dzzU0_imag_train = zeros(N_train,1);


for i = 1:N_train
    
    v_train(i,1) = v_in(zz_in(i),xx_in(i),yy_in(i));
    eta_train(i,1) = eta_in(zz_in(i),xx_in(i),yy_in(i));
    delta_train(i,1) = delta_in(zz_in(i),xx_in(i),yy_in(i));
    
    U0_real_train(i,1) = real(G_3d(zz_in(i),xx_in(i),yy_in(i)));
    U0_imag_train(i,1) = imag(G_3d(zz_in(i),xx_in(i),yy_in(i)));
    
    dxxU0_real_train(i,1) = real(dxxG_3d(zz_in(i),xx_in(i),yy_in(i)));
    dxxU0_imag_train(i,1) = imag(dxxG_3d(zz_in(i),xx_in(i),yy_in(i)));
    
    dzzU0_real_train(i,1) = real(dzzG_3d(zz_in(i),xx_in(i),yy_in(i)));
    dzzU0_imag_train(i,1) = imag(dzzG_3d(zz_in(i),xx_in(i),yy_in(i)));
    
    dyyU0_real_train(i,1) = real(dyyG_3d(zz_in(i),xx_in(i),yy_in(i)));
    dyyU0_imag_train(i,1) = imag(dyyG_3d(zz_in(i),xx_in(i),yy_in(i)));
    
end

v0 = ones(N_train,1)*vv0;
m_train  = 1./(v_train(:)).^2;
m0_train = 1./(v0(:)).^2; 

% save over3d_vti_10Hz_train.mat x_train y_train z_train  m_train m0_train  U0_real_train U0_imag_train  delta_train  eta_train  dxxU0_real_train  dxxU0_imag_train  dzzU0_real_train  dzzU0_imag_train   dyyU0_real_train  dyyU0_imag_train      

% load('over3d_vti_15Hz.mat')
% 
% save over3d_vti_15Hz_test.mat dU_real dU_imag x_coor y_coor z_coor 

