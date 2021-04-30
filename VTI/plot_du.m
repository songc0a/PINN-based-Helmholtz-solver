close all
clear all


load('misfit_l40_N20000.mat');

niter = length(misfit);

load('misfit1_l40.mat');

niter = length(misfit1);

mis = [misfit misfit1(3:end)];
nn = length(mis);

figure;
% semilogy(1:niter, misfit1);
semilogy(1:nn, mis,'LineWidth',1.5);
% hold on
% plot([20000,20000],[50,1e3],'--r','LineWidth',2)
xlabel('Epoch','FontSize',12)
ylabel('Loss','FontSize',12);
set(gca,'FontSize',14)

nz = 101; nx = nz; n = [nz,nx]; N = prod(n);
dx = 25; dz = dx; h  = [dz dx];
z  = [0:n(1)-1]'*h(1)/1000;
x  = [0:n(2)-1]*h(2)/1000;

load('broc.mat')

amp = 0.1;

load('du_real_pred_vti_layer.mat')
load('du_imag_pred_vti_layer.mat')
figure;
pcolor(x,z,reshape(du_real_pred,n));
shading interp
axis ij
colorbar; colormap(broc)
caxis([-amp amp]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
set(gca,'FontSize',14)

% figure;
% pcolor(x,z,reshape(du_imag_pred,n))
% shading interp
% axis ij
% colorbar; colormap(broc)
% caxis([-amp amp]);
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
% set(gca,'FontSize',14)


load('du_imag_star_marsrandom.mat')
load('du_real_star_marsrandom.mat')

% du_real = du_real_star;
% du_imag = du_imag_star;

du_real_star_2d = reshape(du_real_star,n);
du_real_star_2d(1:2,49:51) = 0;
du_real_star = du_real_star_2d(:);


misfit_du_real =norm(du_real_star-du_real_pred)

figure;
pcolor(x,z,du_real_star_2d);
shading interp
axis ij
colorbar; colormap(broc)
caxis([-amp amp]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
set(gca,'FontSize',14)

% figure;
% pcolor(x,z,reshape(du_imag_star,n))
% shading interp
% axis ij
% colorbar; colormap(broc)
% caxis([-amp amp]);
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
% set(gca,'FontSize',14)


figure;
pcolor(x,z,reshape(du_real_star-du_real_pred,n));
shading interp
axis ij
colorbar; colormap(broc)
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
caxis([-amp amp]);
set(gca,'FontSize',14)

% figure;
% pcolor(x,z,reshape(du_imag_star-du_imag_pred,n))
% shading interp
% axis ij
% colorbar; colormap(broc)
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
% caxis([-amp amp]);
% set(gca,'FontSize',14)

% load('m_pred_mid19.mat');
% m_pred(m_pred<0)=0.4;
% 
% v_inv = reshape(sqrt(1./m_pred),n);
% figure;
% pcolor(x,z,real(v_inv));
% shading interp
% axis ij
% colorbar; colormap(broc)
% % xlim([0 2]);ylim([0 2])
% caxis([1.5 4]);
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
% set(gca,'FontSize',14)
% 
% 
% load('delta_pred_mid19.mat');
% delta_inv = reshape(delta_pred,n);
% figure;
% pcolor(x,z,real(delta_inv));
% shading interp
% axis ij
% colorbar; colormap(broc)
% caxis([0 0.15]);
% % xlim([0 2]);ylim([0 2])
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
% set(gca,'FontSize',14)
% 












