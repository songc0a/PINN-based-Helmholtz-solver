
function P = vti_getP(n,npmlz,npmlx,Iz,Ix)
% sampling operator, selects gridpoints from 2D grid:
%
% use:
%   P = getP(n,Iz,Ix);
%
% input:
%   n     - [nz,nx] number of gridpoints in z and x direction
%   Iz,Ix - indices
%
% ouput:
%   P - sparse matrix
%
Nx = n(2) + 2*npmlx-2;
Nz = n(1) + 2*npmlz-2;
I1 = speye(Nz);
I2 = speye(Nx);
P  = kron(I2(npmlx-1+Ix,:),I1(npmlz-1+Iz,:));

