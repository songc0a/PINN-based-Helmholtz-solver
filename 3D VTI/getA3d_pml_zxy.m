function [A] = getA3d_pml_zxy(f,v,h,n,npml)
% 5-point discretization of the acoustic 3D Helmholtz operator with PML
% boundary conditions.
%
% axis: zxy
%
% input:
%   f - frequency [Hz]
%   v - model [m/s]
%   h - [dz,dx,dy] gridspacing in z and x and y direction [m]
%   n - [nz,nx,ny] number of gridpoints in z and x and y direction
%
% ouput:
%   A - sparse matrix

% angular frequency
omega = 2*pi*f/1000;
nz = n(1); ny = n(3); nx = n(2);
dz = h(1); dy = h(3); dx = h(2);
sx = ones(nx,1); sy = ones(ny,1); sz = ones(nz,1);

etax = npml*dx;
etaz = npml*dz;
etay = npml*dy;

mvel = v(1,1,1) ;

cx = -1/(2*etax)*log(10^(-log2(npml/10)-3));
cy = -1/(2*etay)*log(10^(-log2(npml/10)-3));
cz = -1/(2*etaz)*log(10^(-log2(npml/10)-3));
k = omega/mvel;

for ix = 1:npml;
    sigmax = ((ix-npml)*dx/etax)^2*cx;
    sx(ix,1) = 1.0 + 1i*sigmax/k;
    sigmay = ((ix-npml)*dy/etay)^2*cy;
    sy(ix,1) = 1.0 + 1i*sigmay/k;
    sigmaz = ((ix-npml)*dz/etaz)^2*cz;
    sz(ix,1) = 1.0 + 1i*sigmaz/k;    
end

for ix = 1:npml;
    sx(nx+1-ix,1) = sx(ix,1);
    sy(nx+1-ix,1) = sy(ix,1);
    sz(nx+1-ix,1) = sz(ix,1);
end

dx2_inv = 1/dx/dx;
dy2_inv = 1/dy/dy;
dz2_inv = 1/dz/dz;

N = nx*ny*nz;
%% left top
nzeros = 1;

for iy = 1:ny;
    for ix = 1:nx;
        for iz = 1:nz;
            ijk = (iy-1)*nx*nz + (ix-1)*nz + iz;
            if( ix==1 || ix==nx || iy==1 || iy==ny || iz==1 || iz==nz )
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk;
                M(nzeros,1)=  -2*(dx2_inv + dy2_inv + dz2_inv);
                nzeros = nzeros + 1;
            else 
                xtmp = sy(iy,1)*sz(iz,1)/sx(ix,1)/dx/dx;
                ytmp = sx(ix,1)*sz(iz,1)/sy(iy,1)/dy/dy;
                ztmp = sy(iy,1)*sx(ix,1)/sz(iz,1)/dz/dz;
                
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk-nx*nz;
                M(nzeros,1)=ytmp;
                nzeros = nzeros + 1;
                 
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk-nz;
                M(nzeros,1)=xtmp;
                nzeros = nzeros + 1;
                
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk-1;
                M(nzeros,1)=ztmp;
                nzeros = nzeros + 1;
                
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk;
                k = omega/v(iz,ix,iy);
                helm_term = sx(ix,1)*sy(iy,1)*sz(iz,1)*k*k;
                a = -2*(xtmp+ytmp+ztmp) + helm_term;
                M(nzeros,1)=a; 
                nzeros = nzeros + 1;
                
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk+1;
                M(nzeros,1)=ztmp;
                nzeros = nzeros + 1; 
                
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk+nz;
                M(nzeros,1)=xtmp;
                nzeros = nzeros + 1;    
                
                Ti(nzeros,1)=ijk;
                Tj(nzeros,1)=ijk+nx*nz;
                M(nzeros,1)=ytmp;
                nzeros = nzeros + 1;               
                
            end
        end
    end
end

A=sparse(Ti,Tj,M,N,N);

end
