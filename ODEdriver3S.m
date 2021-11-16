%%%%%%%%% HO-POD-DEIM MODEL REDUCTION OF 3D BURGERS %%%%%%%%%

% This code implements the HO-POD-DEIM algorithm from the paper:

 %G. Kirsten. Multilinear POD-DEIM model reduction for 2D and 3D nonlinear 
 %systems of differential equations. pp. 1-25, July 2021. 
 % arXiv: 2103.04343 [math.NA]. To apper in J. Computational Dynamics.
 
% This example is applied to the coupled Burgers equation in three
% dimensions.


%We provide this code without any guarantee that the code is bulletproof as input 
%parameters are not checked for correctness.
%Finally, if the user encounters any problems with the code, the author can be
%contacted via e-mail.


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% LIST OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% nn        % The number of discretization nodes in each direction (x,y,z)
% nsmax     % The number of snapshots considered in the offline phase
% tf        % The right extreme of the timespan [0,tf]
% kappa     % The maximum admissable dimension of the reduced model in all directions
% tau       % The tolerance used for basis truncation
% ode      % A collection of all ODE parameters defined below
% r         % The Reynold's number

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nn = 50;
nsmax =20;
kappa= 40;
r = 10;
tau = 1e-3;
tf = 1;

hx = 1/nn;


%% Finite difference discretization of the coupled 3D burgers equation on a square %%

xx = linspace(0,1,nn+1);
yy = linspace(0,1,nn+1);
zz = xx;

[x,y,z] = meshgrid(xx,yy,zz);

nh=  nn+1;

Nx = nh; dx = 1/(Nx);
e = ones(Nx, 1);
A_1D = spdiags([e, -2*e, e], -1:1, Nx, Nx);
AA = A_1D;
sigma = 1/r;

AA(1,1) = -1;AA(1,2) = 0;AA(2,1) = 1;
AA(end,end) = -1; AA(end-1,end) = 1; AA(end,end-1) = 0;

AA = 1/dx^2*sigma*(AA);

for k = 1:3
    for r = 1:3
        A{k,r} = AA;
    end
end
ode.A = A;

BB = eye(Nx) - diag(ones(Nx-1,1),-1);

CC = spdiags([-1*ones(Nx,1), ones(Nx,1)], [-1,1], Nx, Nx);
CC(1,2) = 0 ; CC(end,end-1) = 0;
for k = 1:3
    for r = 1:3
        B{k,r} = (1/(2*dx))*CC;
    end
end
ode.B = B;



%%% Initial condition
X0{1} = 0.1*sin(2.*pi.*x).*sin(2.*pi.*y).*cos(2.*pi.*z);
X0{2} = 0.1*sin(2.*pi.*x).*cos(2.*pi.*y).*sin(2.*pi.*z);
X0{3} = 0.1*cos(2.*pi.*x).*cos(2.*pi.*y).*sin(2.*pi.*z);

for k = 1:3
    X0{k} = tensor(X0{k},[nn+1,nn+1,nn+1]);
end


ode.X0 = X0;


%% Call the code %%

vanilla.nn = nn;
vanilla.nmax = nsmax;
vanilla.kaps = kappa;
vanilla.tf = tf;
vanilla.tau = tau;
vanilla.ode = ode;

[romtime2] = integrateODEsystem3Sburger(vanilla);




