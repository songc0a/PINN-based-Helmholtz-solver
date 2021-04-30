%%Acoustic VTI modelling
clear all
close all

load('layer_v.mat');
load('layer_eta.mat');
delta = eta;

nz = 101; nx = 101;
n  = [nz nx]; N = n(1)*n(2);

vv = 1.5;

v0 = ones(nz, nx)*vv;
delta0 = zeros(nz, nx);
eta0 = zeros(nz, nx);

dx = 0.025; dz = 0.025;
h  = [dz dx];
z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);

figure;
pcolor(x,z,v);
shading interp
axis ij
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
colormap(jet)
colorbar
caxis([1.5 4.0])
set(gca,'FontSize',14)

figure;
pcolor(x,z,delta);
shading interp
axis ij
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
colormap(jet)
colorbar
caxis([0 0.15])
set(gca,'FontSize',14)

npmlz = 50; npmlx = 50; 
Nz = nz + 2*npmlz;
Nx = nx + 2*npmlx;
NN = (Nx-2)*(Nz-2);
v0_e=extend2d(v0,npmlz,npmlx,Nz,Nx);
delta0_e=extend2d(delta0,npmlz,npmlx,Nz,Nx);
eta0_e=extend2d(eta0,npmlz,npmlx,Nz,Nx);

v_e=extend2d(v,npmlz,npmlx,Nz,Nx);
delta_e=extend2d(delta,npmlz,npmlx,Nz,Nx);
eta_e=extend2d(eta,npmlz,npmlx,Nz,Nx);

f = 5;
Av  = vti_getA(f,v_e,delta_e,eta_e,h,npmlz,npmlx,Nz,Nx);


% source and receiver coordinates
src_x = 51;
src_z = 2;
% src_z = 51;

Ps1 = vti_getP(n,npmlz,npmlx,src_z,src_x);
Ps2 = zeros(size(Ps1));
Ps = [Ps1 Ps2];

Uv  = Av\(Ps'*1000);
U_top = Uv(1:NN,1);

Ph_vti = zeros(n); Ph_vti0 = zeros(n);

for ix = (npmlx+1):(Nx-npmlx);
    for iz = (npmlz+1):(Nz-npmlz);

        Uv_2d(iz-npmlz,ix-npmlx) = Uv((ix-1)*(Nz-2)+iz-1);

    end
end

[zz,xx] = ndgrid(z,x);

% Source location, [m]
xs = (src_x-2)*dx;
zs = (src_z-1)*dz;
% Source frequency, [Hz]

%% ANALYTICAL
% Distance from source to each point in the model
r = @(zz,xx)(zz.^2+xx.^2).^0.5;
% Angular frequency
omega = 1*2*pi*f;
% Wavenumber

K = (omega./vv);

G_2D_analytic = @(zz,xx)0.25i * besselh(0,2,(K) .* r(zz,xx));
G_2D = (G_2D_analytic(zz - zs, xx - xs))*0.65;

[isz,isx]=find(isnan(G_2D));

G_2D(isz,isx) = (G_2D(isz-1,isx) + G_2D(isz+1,isx) + ...
    G_2D(isz,isx-1) + G_2D(isz,isx+1))/4;   

Uv_2d = conj(Uv_2d);
dU_2d = Uv_2d - G_2D;

dU_2d(real(dU_2d)<-0.1) = 0;

figure;
pcolor(x,z,real(Uv_2d));
shading interp
axis ij
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
colormap(jet)
colorbar
caxis([-0.1 0.1])
set(gca,'FontSize',14)

figure;
pcolor(x,z,real(G_2D));
shading interp
axis ij
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
colormap(jet)
colorbar
caxis([-0.1 0.1])
set(gca,'FontSize',14)

figure;
pcolor(x,z,real(dU_2d));
shading interp
axis ij
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
colormap(jet)
colorbar
caxis([-0.05 0.05])
set(gca,'FontSize',14)


xx = repmat(x,nz,1);
zz = repmat(z,nx,1);
x_star = xx(:);
z_star = zz(:);

%%%Training data

x1 = (0:2501-1)*0.001;
z1 = (0:2501-1)'*0.001;
[zz1,xx1] = ndgrid(z1,x1);

G_2D = (G_2D_analytic(zz1 - zs, xx1 - xs))*0.65;

[isz,isx]=find(isnan(G_2D));

G_2D(isz,isx) = (G_2D(isz-1,isx) + G_2D(isz+1,isx) + ...
    G_2D(isz,isx-1) + G_2D(isz,isx+1))/4; 

[G_2D_fx,G_2D_fy]=gradient(G_2D);
[G_2D_fxx,G_2D_fxy]=gradient(G_2D_fx);
[G_2D_fyx,G_2D_fyy]=gradient(G_2D_fy);
dzzG_2D = G_2D_fyy/0.001/0.001;
dxxG_2D = G_2D_fxx/0.001/0.001;


[Xq,Yq] = meshgrid(x1,z1);
v_in = interp2(x,z,v,Xq,Yq);
eta_in = interp2(x,z,eta,Xq,Yq);
delta_in = interp2(x,z,delta,Xq,Yq);

N_train = 2500;

x_train = rand(N_train,1)*2.5 + 0.0;
z_train = rand(N_train,1)*2.5 + 0.0;

xx_in = round(x_train/0.001)+1;
zz_in = round(z_train/0.001)+1;

v_train = zeros(N_train,1);
eta_train = zeros(N_train,1);
delta_train = zeros(N_train,1);
U0_imag_train = zeros(N_train,1);
U0_real_train = zeros(N_train,1);
dxxU0_imag_train = zeros(N_train,1);
dxxU0_real_train = zeros(N_train,1);
dzzU0_real_train = zeros(N_train,1);
dzzU0_imag_train = zeros(N_train,1);

for i = 1:N_train
    
    v_train(i,1) = v_in(zz_in(i),xx_in(i));
    eta_train(i,1) = eta_in(zz_in(i),xx_in(i));
    delta_train(i,1) = delta_in(zz_in(i),xx_in(i));
    
    U0_real_train(i,1) = real(G_2D(zz_in(i),xx_in(i)));
    U0_imag_train(i,1) = imag(G_2D(zz_in(i),xx_in(i)));
    
    dxxU0_real_train(i,1) = real(dxxG_2D(zz_in(i),xx_in(i)));
    dxxU0_imag_train(i,1) = imag(dxxG_2D(zz_in(i),xx_in(i)));
    
    dzzU0_real_train(i,1) = real(dzzG_2D(zz_in(i),xx_in(i)));
    dzzU0_imag_train(i,1) = imag(dzzG_2D(zz_in(i),xx_in(i)));
    
end

dU = full(dU_2d(:)); dU_real = real(dU); dU_imag = imag(dU);
dU(abs(dU)>1)=0;

v0 = ones(N_train,1)*vv;
m_train  = 1./(v_train(:)).^2;
m0_train = 1./(v0(:)).^2; 

save layer_vti_5Hz_test_data.mat x_star z_star dU_real dU_imag 
save layer_vti_5hz_train_data.mat x_train z_train U0_real_train U0_imag_train m_train m0_train delta_train eta_train dxxU0_real_train dxxU0_imag_train dzzU0_real_train dzzU0_imag_train     







