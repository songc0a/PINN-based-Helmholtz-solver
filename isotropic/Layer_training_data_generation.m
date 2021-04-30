clear all 
close all

load('layer_velocity.mat');

n = size(v);
dx = 0.025; dz = 0.025;
nx = n(2); nz = n(1);
h  = [dz dx];

N_train = 2000; %% number of the random points

z_train  = 2.5*rand(N_train,1)+0.0;
x_train  = 2.5*rand(N_train,1)+0.0;

v0 = ones(N_train,1)*1500;

v = v/1000; v0 = v0/1000;

src_z = 2; 
sz = (src_z-1)*dz; %% depth of the source
src_x = 51; 
sx = (src_x-1)*dx; %% horizontal location of the source

z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);
[X,Y] = meshgrid(x,z);
x1 = [0:2501-1]*0.001;
z1 = [0:2501-1]'*0.001;
[Xq,Yq] = meshgrid(x1,z1);
v_in = interp2(x,z,v,Xq,Yq); %% velocity interpolation

xx = rand(N_train,1)*2.45 + 0.05; %% random x coordinate values
zz = rand(N_train,1)*2.45 + 0.05; %% random z coordinate values

xx_in = round(xx/0.001)+1;
zz_in = round(zz/0.001)+1;

v_train = zeros(N_train,1);
U0_imag_train = zeros(N_train,1);
U0_real_train = zeros(N_train,1);

f = 5.0; %% frequency
%% ANALYTICAL Solution (Background wavefield)
% Distance from source to each point in the model
r = @(zz,xx)(zz.^2+xx.^2).^0.5;
% Angular frequency
omega = 1*2*pi*f;
% Wavenumber
vv = 1.5;
K = (omega./vv);
x = (0:2501-1)*0.001;
z = (0:2501-1)'*0.001;
[zz1,xx1] = ndgrid(z,x);

G_2D_analytic = @(zz,xx)0.25i * besselh(0,2,(K) .* r(zz,xx));

for i = 1:N_train
    
    G_2D = (G_2D_analytic(zz(i) - sz, xx(i) - sx))*7.7;
    
    v_train(i,1) = v_in(zz_in(i),xx_in(i));
    
    U0_real_train(i,1) = real(G_2D);
    U0_imag_train(i,1) = imag(G_2D);

end

m_train = 1./v_train.^2;
m0_train = ([(1./(v0).^2)]);

x_train = xx; 
z_train = zz;

save layer_5Hz_train_data.mat U0_real_train U0_imag_train x_train z_train m_train m0_train

%% Numerical results

z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);

[zz,xx] = ndgrid(z,x);

x_star = xx(:); 
z_star = zz(:); 

npmlz = 60; npmlx = npmlz;
Nz = nz + 2*npmlz;
Nx = nx + 2*npmlx;

v0 = ones(n)*1.500;

v_e=extend2d(v,npmlz,npmlx,Nz,Nx);

Ps1 = getP_H(n,npmlz,npmlx,src_z,src_x);
Ps1 = Ps1'*12000;

[o,d,n] = grid2odn(z,x);
n=[n,1];

nb = [npmlz  npmlx 0];
n  = n + 2*nb;

f = 5; omega = 2*pi*f;
A = Helm2D((omega)./v_e(:),o,d,n,nb);
U  = A\Ps1;

U_2D = reshape(full(U),[nz+2*npmlz,nx+2*npmlx]);
U_2d = U_2D(npmlz+1:end-npmlz,npmlx+1:end-npmlx);

G_2D = (G_2D_analytic(zz - sz, xx - sx))*7.7;

G_2D(src_z,src_x) = (G_2D(src_z-1,src_x) + G_2D(src_z+1,src_x) + G_2D(src_z,src_x-1) + G_2D(src_z,src_x+1))/4;
dU_2d = U_2d-G_2D;

dU_real_star = real(dU_2d(:));
dU_imag_star = imag(dU_2d(:));

dU_real_star(abs(dU_real_star)>2)=0;

save layer_5Hz_test_data.mat x_star z_star dU_real_star dU_imag_star

