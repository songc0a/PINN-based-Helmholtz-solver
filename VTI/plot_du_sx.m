close all
clear all

load('misfit_adam_sx.mat');

load('misfit_lbfgs_sx.mat');

mis = [misfit misfit1];
nn = length(mis);

figure;
semilogy(1:nn, mis,'LineWidth',1.5);
xlabel('Epoch','FontSize',12)
ylabel('Loss','FontSize',12);
set(gca,'FontSize',14)

nz = 101; nx = 101; n = [nz,nx];
dx = 25; dz = dx; h  = [dz dx];
z  = [0:n(1)-1]'*h(1)/1000;
x  = [0:n(2)-1]*h(2)/1000;

amp = 0.5;

load('du_imag_pred_sx.mat')
load('du_real_pred_sx.mat')

is = 7;

du_real_pred = du_real_pred( ((is-1)*nz*nx+1) : (is*nz*nx) ,1);
du_imag_pred = du_imag_pred( ((is-1)*nz*nx+1) : (is*nz*nx) ,1);

figure;
pcolor(x,z,reshape(du_real_pred,n));
shading interp
axis ij
colorbar; colormap(jet)
% xlim([0 2]);ylim([0 2])
caxis([-amp amp]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
set(gca,'FontSize',14)

% title('ML scattered wavefield');
figure;
pcolor(x,z,reshape(du_imag_pred,n))
shading interp
axis ij
colorbar; colormap(jet)
% xlim([0 2]);ylim([0 2])
caxis([-amp amp]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
% title('ML scattered wavefield');
set(gca,'FontSize',14)

load('du_imag_star_G2D.mat')
load('du_real_star_G2D.mat')

du_real_star(abs(du_real_star )>2)=0;

du_real_star = du_real_star( ((is-1)*nz*nx+1) : (is*nz*nx) ,1);
du_imag_star = du_imag_star( ((is-1)*nz*nx+1) : (is*nz*nx) ,1);


figure;
pcolor(x,z,reshape(du_real_star,n));
shading interp
axis ij
colorbar; colormap(jet)
% xlim([0 2]);ylim([0 2])
caxis([-amp amp]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
set(gca,'FontSize',14)

% title('True scattered wavefield');
figure;
pcolor(x,z,reshape(du_imag_star,n))
shading interp
axis ij
colorbar; colormap(jet)
% xlim([0 2]);ylim([0 2])
caxis([-amp amp]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
% title('True scattered wavefield');
set(gca,'FontSize',14)

% amp = 0.01;
figure;
pcolor(x,z,reshape(du_real_star-du_real_pred,n));
shading interp
axis ij
colorbar; colormap(jet)
% xlim([0 2]);ylim([0 2])
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
% title('Wavefield difference');
caxis([-amp amp]);
set(gca,'FontSize',14)

figure;
pcolor(x,z,reshape(du_imag_star-du_imag_pred,n))
shading interp
axis ij
colorbar; colormap(jet)
% xlim([0 2]);ylim([0 2])
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
% title('Wavefield difference');
caxis([-amp amp]);
set(gca,'FontSize',14)

trace = 1 + is*10;

du_real_pred2d = reshape(du_real_pred,n);
du_imag_pred2d = reshape(du_imag_pred,n);


du_real_star2d = reshape(du_real_star,n);
du_imag_star2d = reshape(du_imag_star,n);

du_real_pred2d_p = du_real_pred2d(:,trace);
du_imag_pred2d_p = du_imag_pred2d(:,trace);

du_real_star2d_p = du_real_star2d(:,trace);
du_imag_star2d_p = du_imag_star2d(:,trace);


figure;
plot(du_real_star2d_p,z,'r-','LineWidth',1.5)
hold on
plot(du_real_pred2d_p,z,'k--','LineWidth',1.5)
hold on
axis ij
ylabel('Depth (km)','FontSize',12)
xlabel('Amplitude','FontSize',12);
legend('Numerical','PINN')
set(gca,'FontSize',14)
xlim([-1 1])

figure;
plot(du_imag_star2d_p,z,'r-','LineWidth',1.5)
hold on
plot(du_imag_pred2d_p,z,'k--','LineWidth',1.5)
hold on
axis ij
ylabel('Depth (km)','FontSize',12)
xlabel('Amplitude','FontSize',12);
legend('Numerical','PINN')
set(gca,'FontSize',14)
xlim([-1 1])

% load('du_real_xx_pred_mid.mat')
% load('du_real_yy_pred_mid.mat')
% 
% 
% figure;
% pcolor(x,z,reshape(du_real_xx_pred,n));
% shading interp
% axis ij
% colorbar; colormap(jet)
% % xlim([0 2]);ylim([0 2])
% % caxis([-amp amp]);
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
% set(gca,'FontSize',14)
% 
% % title('ML scattered wavefield');
% figure;
% pcolor(x,z,reshape(du_real_yy_pred,n))
% shading interp
% axis ij
% colorbar; colormap(jet)
% % xlim([0 2]);ylim([0 2])
% % caxis([-amp amp]);
% xlabel('Distance (km)','FontSize',12)
% ylabel('Depth (km)','FontSize',12);
% % title('ML scattered wavefield');
% set(gca,'FontSize',14)
