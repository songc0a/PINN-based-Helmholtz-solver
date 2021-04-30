function [A] = vti_getA(f,v,delta,theta,h,npmlz,npmlx,Nz,Nx)
% 5-point discretization of the 2D Helmholtz operator with PML
% boundary conditions.
%
% use:
%   A = vti_getA(f,m,epsilon,theta,h,n);
%
% input:
%   f - frequency [Hz]
%   v - model [m/s]
%   delta - anisotropic parameter delta
%   theta - anisotropic parameter theta
%   h - [dz,dx] gridspacing in z and x direction [m]
%   n - [nz,nx] number of gridpoints in z and x direction
%
% ouput:
%   A - sparse matrix

% angular frequency
omega = 2*pi*f;
dz = h(1); dx = h(2);
etax = npmlx*dx;
etaz = npmlz*dz;
eps = 0.01;
vmax = max(max(v));
c1 = -3*log(eps)*vmax/(etaz*omega);
c2 = -3*log(eps)*vmax/(etax*omega);

gz = zeros(Nz,1);
for iz = 1:Nz;
    if iz<npmlz+1;
        gz(iz) = (c1*(npmlz-iz)*dz/etaz)^2;
    else if iz>=Nz-npmlz+1;
            gz(iz) = (c1*(iz-(Nz-npmlz+1))*dz/etaz)^2;
        else gz(iz) = 0;
        end
    end
end
sz = 1./(1+1i*gz);
gx = zeros(Nx,1);
for ix = 1:Nx;
    if ix<npmlx+1;
        gx(ix) = (c2*(npmlx-ix)*dx/etax)^2;
    else if ix>=Nx-npmlx+1;
            gx(ix) = (c2*(ix-(Nx-npmlx+1))*dx/etax)^2;
        else gx(ix) = 0;
        end
    end
end
sx = 1./(1+1i*gx);
NN = 2*(Nz-2)*(Nx-2);
%% left top
count = 1;

for ix = 2:Nx-1;
    for iz = 2:Nz-1;
        index = (ix-2)*(Nz-2)+iz-1;       
        %second derivative with respect to x 
        cent = 0 + 1i*0;
        neib = (sx(ix)/sz(iz)+sx(ix-1)/sz(iz))/(2*dx*dx);
        cent = cent - neib;
        if ix~=2;
            Ti(count,1)=index;
            Tj(count,1)=index-(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end
        neib = (sx(ix)/sz(iz)+sx(ix+1)/sz(iz))/(2*dx*dx);
        cent = cent - neib;
        if ix~=Nx-1;
            Ti(count,1)=index;
            Tj(count,1)=index+(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end  
        
        %second derivative with respect to z
        neib = (sz(iz)/sx(ix)+sz(iz-1)/sx(ix))/(2*dz*dz)/(1+2*delta(iz,ix));
        cent = cent - neib;
        if iz~=2;
            Ti(count,1)=index;
            Tj(count,1)=index-1;
            M(count,1)=neib;
            count = count+1;
        end
        neib = (sz(iz)/sx(ix)+sz(iz+1)/sx(ix))/(2*dz*dz)/(1+2*delta(iz,ix));
        cent = cent - neib;
        if iz~=Nz-1;
            Ti(count,1)=index;
            Tj(count,1)=index+1;
            M(count,1)=neib;
            count = count+1;
        end        
        %center
        cent = cent + omega^2/v(iz,ix)^2/(sz(iz)*sx(ix));
        Ti(count,1)=index;
        Tj(count,1)=index;
        M(count,1)=cent;
        count = count+1;   
 %%right top       
        cent = 0 + 1i*0;
        %second derivative with respect to x   
        neib = (sx(ix)/sz(iz)+sx(ix-1)/sz(iz))/(2*dx*dx);
        cent = cent - neib;
        if ix~=2;
            Ti(count,1)=index;
            Tj(count,1)=index+(Nx-2)*(Nz-2)-(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end
        neib = (sx(ix)/sz(iz)+sx(ix+1)/sz(iz))/(2*dx*dx);
        cent = cent - neib;
        if ix~=Nx-1;
            Ti(count,1)=index;
            Tj(count,1)=index+(Nx-2)*(Nz-2)+(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end       
         %center
        %cent = cent + omega^2/v(iz,ix)^2/(sz(iz)*sx(ix));
        Ti(count,1)=index;
        Tj(count,1)=index+(Nx-2)*(Nz-2);
        M(count,1)=cent;
        count = count+1;       
        
    end
end

for ix = 2:Nx-1;
    for iz = 2:Nz-1;
        index = (ix-2)*(Nz-2)+iz-1+(Nx-2)*(Nz-2);  
 %%left bottom
        %second derivative with respect to x 
        cent = 0 + 1i*0;
        neib = (sx(ix)/sz(iz)+sx(ix-1)/sz(iz))/(2*dx*dx)*(2*theta(iz,ix));
        cent = cent - neib;
        if ix~=2;
            Ti(count,1)=index;
            Tj(count,1)=index-(Nx-2)*(Nz-2)-(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end
        neib = (sx(ix)/sz(iz)+sx(ix+1)/sz(iz))/(2*dx*dx)*(2*theta(iz,ix));
        cent = cent - neib;
        if ix~=Nx-1;
            Ti(count,1)=index;
            Tj(count,1)=index-(Nx-2)*(Nz-2)+(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end         
        %center
       %cent = cent + omega^2/v(iz,ix)^2/(sz(iz)*sx(ix));
        Ti(count,1)=index;
        Tj(count,1)=index-(Nx-2)*(Nz-2);
        M(count,1)=cent;
        count = count+1;       
      
 %%right bottom       
        cent = 0 + 1i*0;
        %second derivative with respect to x   
        neib = (sx(ix)/sz(iz)+sx(ix-1)/sz(iz))/(2*dx*dx)*(2*theta(iz,ix));
        cent = cent - neib;
        if ix~=2;
            Ti(count,1)=index;
            Tj(count,1)=index-(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end
        neib = (sx(ix)/sz(iz)+sx(ix+1)/sz(iz))/(2*dx*dx)*(2*theta(iz,ix));
        cent = cent - neib;
        if ix~=Nx-1;
            Ti(count,1)=index;
            Tj(count,1)=index+(Nz-2);
            M(count,1)=neib;
            count = count+1;
        end       
         %center
        cent = cent + omega^2/v(iz,ix)^2/(sz(iz)*sx(ix));
        Ti(count,1)=index;
        Tj(count,1)=index;
        M(count,1)=cent;
        count = count+1;       
        
    end
end

A=sparse(Ti,Tj,M,NN,NN);

end
