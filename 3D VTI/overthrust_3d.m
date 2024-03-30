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
v0 = ones(size(v))*2.286;

sigma = 1;
v = imgaussfilt3(v,sigma);

n = size(v);
nz = n(1); ny = n(3); nx = n(2);

npml = 20;
v = extend3d(v,npml,nz,nx,ny); v0 = extend3d(v0,npml,nz,nx,ny); 
v_true = v(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
v_ini = v0(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);

dx = 0.025; dy = 0.025; dz = 0.025;
clim = [1.500 5.500];

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(v_true, dz,dx,dy, clim, [50*dx], [50*dy], [55*dz], [], ...
    'attributeName', 'V (km/s)', ...
    'filtCoef', 0.4, ...
    'startTime', 0.1);

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(v_true, dz,dx,dy, clim, [1*dx], [1*dy], [0*dz], []);

n = size(v);
nz = n(1); ny = n(3); nx = n(2);

src_x = round(nx/2);
src_y = round(ny/2);
src_z = npml+2;

ifre = 10;
A = getA3d_pml_zxy(ifre,v,h,n,npml);
A0 = getA3d_pml_zxy(ifre,v0,h,n,npml);

Ps = getP3d_zxy(n,src_z,src_x,src_y); Ps = Ps'/2;

U  = A\(Ps);
U0  = A0\(Ps);

% Nx = nx + 2*npml; Nz = nz + 2*npml; Ny = ny + 2*npml;

U_3D = (reshape(full(U),n));
U0_3D = (reshape(full(U0),n));

U_3d = U_3D(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
U0_3d = U0_3D(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
dU_3d = U_3d - U0_3d ;

amp = max(max(max(real(U_3d))));
clim = [-2.5 2.5];

%  clim = [-10e6 10e6];
% dx = 25/1000; dy = 25/1000; dz = 25/1000;

% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(real(U_3d), dx,dy,dz, clim, [40*dx], [40*dy], [20*dz], [], ...
%     'attributeName', 'Amplitude', ...
%     'filtCoef', 0.4, ...
%     'startTime', 0.1);
% 
% 
% 
% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(real(U0_3d), dx,dy,dz, clim, [40*dx], [40*dy], [20*dz], [], ...
%     'attributeName', 'Amplitude', ...
%     'filtCoef', 0.4, ...
%     'startTime', 0.1);


figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(real(U_3d), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');


% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(imag(U_3d), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(real(U0_3d), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');


% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(imag(U0_3d), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');


n = size(U_3d);
nz = n(1); ny = n(3); nx = n(2);

z = [0:n(1)-1]'*h(1);
y = [0:n(2)-1]*h(2);
x = [0:n(3)-1]*h(3);

xx = repmat(x,nz,1); 
zz = repmat(z,nx,1); 
yy = repmat(y,nx*nz,1);

x1 = xx(:); x_coor = repmat(x1,ny,1); 
z1 = zz(:); z_coor = repmat(z1,ny,1);
y_coor = yy(:);


% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(imag(reshape(dU_3d,n)), dx,dy,dz, clim, [40*dx], [40*dy], [20*dz], [], ...
%     'attributeName', 'Amplitude', ...
%     'filtCoef', 0.4, ...
%     'startTime', 0.1);

figure;
set(gcf, 'position', [ 208   233   760   517]);
subplot('Position', [0.09 0.13 0.86 0.86]);
stpShow3DVolume(real(reshape(dU_3d,n)), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');

% figure;
% set(gcf, 'position', [ 208   233   760   517]);
% subplot('Position', [0.09 0.13 0.86 0.86]);
% stpShow3DVolume(imag(reshape(dU_3d,n)), dx,dy,dz, clim, [1*dx], [1*dy], [0*dz], [], 'attributeName', 'Amplitude');

% v = v(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
% v0 = v0(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);


% U0 = U0_3d(:); U0_real = real(U0); U0_imag = imag(U0);
% dU = dU_3d(:); dU_real = real(dU); dU_imag = imag(dU);
% v = v(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
% v0 = v0(npml+1:end-npml,npml+1:end-npml,npml+1:end-npml);
% m  = 1./(v(:)).^2; m0 = 1./(v0(:)).^2; 
% 
% save over3d_10Hz_normalized_32_sigma1_simple_div2.mat dU_real dU_imag U0_real U0_imag x_coor z_coor y_coor m m0






