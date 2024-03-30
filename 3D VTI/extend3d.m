function Ve=extend3d(V,npml,nz,nx,ny);

Nx = nx + 2*npml;
Nz = nz + 2*npml;
Ny = ny + 2*npml;

Ve=zeros(Nz,Nx,Ny);

for iy=1:ny
    for ix = 1:nx
        for iz = 1:nz
            Ve(iz+npml,ix+npml,iy+npml) = V(iz,ix,iy);
        end
    end
end
% 
for iy=1:Ny
    for ix = 1:Nx
        for iz = 1:npml
            Ve(iz,ix,iy) = Ve(npml+1,ix,iy);
            Ve(Nz-iz+1,ix,iy) = Ve(Nz-npml,ix,iy);
        end
    end
end
for iy=1:Ny
    for ix = 1:npml
        for iz = 1:Nz
            Ve(iz,ix,iy) = Ve(iz,npml+1,iy);
            Ve(iz,Nx-ix+1,iy) = Ve(iz,Nx-npml,iy);
        end
    end
end

for iy=1:npml
    for ix = 1:Nx
        for iz = 1:Nz
            Ve(iz,ix,iy) = Ve(iz,ix,npml+1);
            Ve(iz,ix,Ny-iy+1) = Ve(iz,ix,Ny-npml);
        end
    end
end

end

