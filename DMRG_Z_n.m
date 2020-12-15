%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DMRG for the 1D Heisenberg Model
% After S.R.White, Phys. Rev. Lett. 69, 2863 (1992)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
warning off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of states kept m
m = 2^8;
% Number of iterations. Final lattice size is 2*Niter + 2
NIter = 4;
% exact energy, for comparison
ExactEnergy = -log(2) + 0.25;
J = 0; % Interaction intensity.
h = 1-J; % Magnetic field strength
dim_local = 3; % 3 --> Z_3 model / 2 --> Z_2 model
% Header:
fprintf('Iter\tEnergy\t\tBondEnergy\tEnergyError\tTrunc\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intialize local operators
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega = exp(1i*2*pi/dim_local);
I = eye(dim_local);
A1 = ones(dim_local - 1,1);
tau = diag(A1,1);
tau(dim_local,1) = 1;
A2 = zeros(dim_local,1);
for ii = 1 :dim_local
A2(ii) = omega^(ii-1);
end
sigma = diag(A2);

A3 = 0:(dim_local - 1);
n_dn = diag(A3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if dim_local == 2
case_real = 1;
prf = 1/2;
else
case_real = 0;
prf = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial blocks
% We assume reflection symmetry so we only need 1 block
% The operator acts on the inner-most site of the block
% +---------+ +---------+
% | *| |* |
% +---------+ +---------+
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BlockTau = tau;
BlockSigma = sigma;
BlockI = I;
BlockH = -prf*( h*sigma + h'*sigma');
H_int = - J*kron(BlockTau', BlockTau);
H_super = kron(BlockH, BlockI) + kron(BlockI, BlockH) ...
+ prf*(H_int + H_int');
[~, Energy] = eigs(H_super,1,'SA');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin main iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for l = 2:NIter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the 2m-dimensional operators for the block + site
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BlockH = kron(BlockH, I) ...
- prf*(J*kron(BlockTau',tau) + J'*kron(BlockTau,tau')) ...
- prf*(h*kron(BlockI,sigma) + h'*kron(BlockI,sigma'));
BlockTau = kron(BlockI, tau);
BlockSigma = kron(BlockI, sigma);
BlockI = kron(BlockI, I);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HAMILTONIAN MATRIX for superblock
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H_int = - J*kron(BlockTau', BlockTau);

H_super = kron(BlockH, BlockI) + kron(BlockI, BlockH) ...
+ prf*(H_int + H_int');
H_super = 0.5 * (H_super + H_super'); % ensure H is symmetric
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagonalizing the Hamiltonian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LastEnergy = Energy;
opts.disp = 0; % disable diagnostic information in eigs
opts.issym = 1;
opts.real = 1;
[Psi, Energy] = eigs(H_super,1,'SA', opts);
EnergyPerBond = (Energy - LastEnergy) / 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Form the reduced density matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[nr,nc]=size(Psi);
Dim = sqrt(nr);
PsiMatrix = reshape(Psi, Dim, Dim);
Rho = PsiMatrix * PsiMatrix';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diagonalize the density matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[V, D] = eig(Rho);
[D, Index] = sort(diag(D), 'descend'); % sort eigenvalues descending
V = V(:,Index); % sort eigenvectors the same way
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct the truncation operator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NKeep = min(size(D, 1), m);
T = V(:, 1:NKeep);
TruncationError = 1 - sum(D(1:NKeep));
fprintf('%d\t%f\t%f\t%f\t%f\n',l, Energy, EnergyPerBond, ...
ExactEnergy-EnergyPerBond, TruncationError);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transform the block operators into the truncated basis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BlockH = T'*BlockH*T;
BlockTau = T'*BlockTau*T;
BlockSigma = T'*BlockSigma*T;
BlockI = T'*BlockI*T;
end