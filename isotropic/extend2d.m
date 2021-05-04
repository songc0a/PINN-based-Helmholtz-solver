function Ve=extend2d(V,npmlz,npmlx,Nz,Nx)

Ve=zeros(Nz,Nx);

Ve(npmlz+1:Nz-npmlz,npmlx+1:Nx-npmlx)=V;

for ii=1:npmlz
    Ve(ii,npmlx+1:Nx-npmlx)=V(1,:);
    Ve(Nz+1-ii,npmlx+1:Nx-npmlx)=V(Nz-2*npmlz,:);
end

for ii=1:npmlx
    Ve(npmlz+1:Nz-npmlz,ii)=V(:,1);
    Ve(npmlz+1:Nz-npmlz,Nx+1-ii)=V(:,Nx-2*npmlx);
end

for ix=1:npmlx
    for iz=1:npmlz
        Ve(iz,ix)=V(1,1);
        Ve(iz,Nx-npmlx+ix)=V(1,Nx-2*npmlx);
        Ve(Nz-npmlz+iz,ix)=V(Nz-2*npmlz,1);
        Ve(Nz-npmlz+iz,Nx-npmlx+ix)=V(Nz-2*npmlz,Nx-2*npmlx);
    end
end
        