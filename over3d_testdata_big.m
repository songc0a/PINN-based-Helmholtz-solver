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

vv = reshape(v_true,n); v = vv(1:2:120,21:4:260,1:3:180);

sigma = 1;
v = imgaussfilt3(v,sigma);

clim = [2.500 3.500];
figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(v, dz,dx,dy, clim, [50*dx], [50*dy], [50*dz], [], ...
    'attributeName', 'V (km/s)', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(v, dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'V (km/s)');

str =['eta_1_true_3D'];
filename=['' str '.bin'];
fid=fopen(filename,'rb');
eta_true=fread(fid,[nz*ny*nx,1],'float');
fclose(fid);

vv = reshape(eta_true,n); eta = vv(1:2:120,21:4:260,1:3:180);
eta = imgaussfilt3(eta,sigma);

clim = [0.000 0.100];
figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(eta, dz,dx,dy, clim, [50*dx], [50*dy], [50*dz], [], ...
    'attributeName', 'Amplitude', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(eta, dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');

str =['del_3_true_3D'];
filename=['' str '.bin'];
fid=fopen(filename,'rb');
delta_true=fread(fid,[nz*ny*nx,1],'float');
fclose(fid);

vv = reshape(delta_true,n); delta = vv(1:2:120,21:4:260,1:3:180);
delta = imgaussfilt3(delta,sigma);

clim = [0.000 0.0500];
figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(delta, dz,dx,dy, clim, [50*dx], [50*dy], [50*dz], [], ...
    'attributeName', 'Amplitude', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(delta, dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');

n = size(v);
nz = n(1); ny = n(3); nx = n(2);

npml = 20;
v = extend3d(v,npml,nz,nx,ny); 
eta = extend3d(eta,npml,nz,nx,ny); 
delta = extend3d(delta,npml,nz,nx,ny); 

n = size(v); vv0 = 3.2;
nz = n(1); ny = n(3); nx = n(2);
v0 = ones(nz,nx,ny)*vv0;

src_x = (nx-1)/2;
src_y = (ny-1)/2;
src_z = (nz-1)/2;


ifre = 15;

omega = 1*2*pi*ifre;
A = getA3d_pml_zxy(ifre,v,h,n,npml);
A0 = getA3d_pml_zxy(ifre,v0,h,n,npml);

Ps = getP3d_zxy(n,src_z,src_x,src_y)/10; Ps = Ps';

U  = A\(Ps);
U0  = A0\(Ps);

U_3D = (reshape(full(U),n));
U0_3D = (reshape(full(U0),n));

U_3d = U_3D(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
U0_3d = U0_3D(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
dU_3d = U_3d - U0_3d ;

U_3D = (reshape(full(U),n));
U0_3D = (reshape(full(U0),n));

% xs = (src_x-1-npml)*dx;
% ys = (src_y-1-npml)*dy;
% zs = (src_z-1-npml)*dz;
% nn = size(U0_3d);
% 
% 
% h  = [dz dx dy];
% n = size(U0_3d);N = prod(n);
% nz = n(1); ny = n(3); nx = n(2);
% 
% z = [0:n(1)-1]'*h(1);
% y = [0:n(2)-1]*h(2);
% x = [0:n(3)-1]*h(3);
% 
% [zz,xx,yy] = ndgrid(z,x,y);
%  
% r = @(zz,xx,yy)(zz.^2+xx.^2+yy.^2).^0.5;
% % Wavenumber
% K = (omega./vv0);
% % For 3D case
% G3D = @(zz,xx,yy)exp(1i*K.*r(zz,xx,yy))./r(zz,xx,yy);
% G_3d = G3D(zz - zs, xx - xs, yy - ys)/7.75;
% 
% src_x = src_x-npml;
% src_y = src_y-npml;
% src_z = src_z-npml;
% G_3d(src_z,src_x,src_y) = (G_3d(src_z-1,src_x,src_y) + G_3d(src_z+1,src_x,src_y) + G_3d(src_z,src_x-1,src_y) +...
%                           G_3d(src_z,src_x+1,src_y) + G_3d(src_z,src_x,src_y-1) + G_3d(src_z,src_x,src_y+1))/6;   
% 
% G_3d = -G_3d;
% dU_3d1 = U_3d - G_3d ;
% 
% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(real(dU_3d1), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');
% 
% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(real(dU_3d1),dz,dx,dy, clim, [30*dx], [30*dy], [35*dz], [], ...
%     'attributeName', 'V (km/s)', ...
%     'filtCoef', 0.4, ...
%     'startTime', 0.1);

clim = [-0.5 0.5];

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(real(dU_3d), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(real(dU_3d),dz,dx,dy, clim, [30*dx], [30*dy], [35*dz], [], ...
    'attributeName', 'V (km/s)', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(imag(U0_3d), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');
% 
% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(imag(G_3d), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');


v = v(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
eta = eta(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
delta = delta(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
v0 = v0(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);


U0 = U0_3d(:); U0_real = real(U0); U0_imag = imag(U0);
% U0 = G_3D(:); U0_real = real(U0); U0_imag = imag(U0);
dU = dU_3d(:); dU_real = real(dU); dU_imag = imag(dU);
m  = 1./(v(:)).^2; m0 = 1./(v0(:)).^2; 
eta = eta(:); delta = delta(:); 

[U0_3d_fx,U0_3d_fy,U0_3d_fz]=gradient(U0_3d, dx);
[U0_3d_fxx,U0_3d_fxy,U0_3d_fxz]=gradient(U0_3d_fx,dx);
[U0_3d_fxy,U0_3d_fyy,U0_3d_fzy]=gradient(U0_3d_fy,dy);
[U0_3d_fxz,U0_3d_fyz,U0_3d_fzz]=gradient(U0_3d_fz,dz);

U0 = U0_3d(:); U0_real = real(U0); U0_imag = imag(U0);
dxxU0 = full(U0_3d_fxx(:)); dxxU0_real = real(dxxU0); dxxU0_imag = imag(dxxU0);
dyyU0 = full(U0_3d_fyy(:)); dyyU0_real = real(dyyU0); dyyU0_imag = imag(dyyU0);
dzzU0 = full(U0_3d_fzz(:)); dzzU0_real = real(dzzU0); dzzU0_imag = imag(dzzU0);

nn = size(U0_3d);
z = [0:nn(1)-1]'*dz;
y = [0:nn(2)-1]*dy;
x = [0:nn(3)-1]*dx;

[zz,xx,yy] = ndgrid(z,x,y);

x_coor = xx(:);
z_coor = zz(:);
y_coor = yy(:);

save over3d_vti_15Hz_test_big.mat dU_real dU_imag x_coor y_coor z_coor 




