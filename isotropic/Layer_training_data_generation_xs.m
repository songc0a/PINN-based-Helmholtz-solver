clear all 
close all

load('layer_velocity.mat');

n = size(v);
dx = 0.025; dz = 0.025;
nx = n(2); nz = n(1);
h  = [dz dx];

N_train = 40000; %% number of the random points

z_train  = 2.5*rand(N_train,1)+0.0;
x_train  = 2.5*rand(N_train,1)+0.0;

v0 = ones(N_train,1)*1500;

v = v/1000; v0 = v0/1000;

src_z = 2; 
sz = (src_z-1)*dz; %% depth of the source
sx = 2.45*rand(N_train,1)+0.05; %% random horizontal source location coordinate values

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

for is = 1:N_train
    
    G_2D = (G_2D_analytic(zz(is) - sz, xx(is) - sx(is)))*7.7;
    
    v_train(is,1) = v_in(zz_in(is),xx_in(is));
    
    U0_real_train(is,1) = real(G_2D);
    U0_imag_train(is,1) = imag(G_2D);

end

m_train = 1./v_train.^2;
m0_train = ([(1./(v0).^2)]);

x_train = xx; 
z_train = zz;
sx_train = sx;

save layer_5Hz_train_data_xs.mat U0_real_train U0_imag_train x_train sx_train z_train m_train m0_train

%% Numerical results

z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);
sx = (10:10:90)*dx; ns = length(sx);

[zz,xx] = ndgrid(z,x);
sx = repmat(sx,nx*nz,1);

x1 = xx(:); x_star = (repmat(x1,ns,1)); 
z1 = zz(:); z_star = (repmat(z1,ns,1));
sx_star = sx(:);

npmlz = 60; npmlx = npmlz;
Nz = nz + 2*npmlz;
Nx = nx + 2*npmlx;

v0 = ones(n)*1.500;

v_e=extend2d(v,npmlz,npmlx,Nz,Nx);

src_x = 11:10:91;
src_z = 2;

Ps1 = getP_H(n,npmlz,npmlx,src_z,src_x);
Ps1 = Ps1'*12000;

[o,d,n] = grid2odn(z,x);
n=[n,1];

nb = [npmlz  npmlx 0];
n  = n + 2*nb;

f = 5; omega = 2*pi*f;
A = Helm2D((omega)./v_e(:),o,d,n,nb);
U  = A\Ps1;

for is = 1:ns

    U_2D = reshape(full(U(:,is)),[nz+2*npmlz,nx+2*npmlx]);
    U_2d = U_2D(npmlz+1:end-npmlz,npmlx+1:end-npmlx);
    
    xs = (src_x(is)-1)*dx;
    zs = (src_z-1)*dz;

    G_2D = (G_2D_analytic(zz - zs, xx - xs))*7.7;  
    
    G_2D(src_z,src_x(is)) = (G_2D(src_z-1,src_x(is)) + G_2D(src_z+1,src_x(is)) + G_2D(src_z,src_x(is)-1) + G_2D(src_z,src_x(is)+1))/4;
    dU_2d = U_2d-G_2D;
    
    dU_real_star( ((is-1)*nz*nx+1) : (is*nz*nx) ,1) = real(dU_2d(:));
    dU_imag_star( ((is-1)*nz*nx+1) : (is*nz*nx) ,1) = imag(dU_2d(:));
    
end

dU_real_star(abs(dU_real_star)>2)=0;

save layer_5Hz_test_data_xs.mat x_star sx_star z_star dU_real_star dU_imag_star

