function [Ynew,Vbasis] = integrateODEsystem3Sburger(firstnotseparatein)

%%%%%%%%% HO-POD-DEIM MODEL REDUCTION OF 3D BURGERS %%%%%%%%%

% This code implements the HO-POD-DEIM algorithm from the paper:

%G. Kirsten. Multilinear POD-DEIM model reduction for 2D and 3D nonlinear
%systems of differential equations. pp. 1-25, July 2021.
% arXiv: 2103.04343 [math.NA]. To apper in J. Computational Dynamics.

% This example is applied to the coupled Burgers equation in three
% dimensions.


% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
% FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
% COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
% IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
% CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% LIST OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% nn        % The number of discretization nodes in each direction (x,y,z)
% nsmax     % The number of snapshots considered in the offline phase
% tf        % The right extreme of the timespan [0,tf]
% kappa     % The maximum admissable dimension of the reduced model in all directions
% tau       % The tolerance used for basis truncation
% ode       % A collection of all ODE parameters defined below
% r         % The Reynold's number

% Given the basis vectors and the set of core tensors, the approximate
% solution can e.g., be lifted as follows:

%     for k = 1:3
%         Xapprox{k} = Ynew{k};

%         for r = 1:3
%             
%             Xapprox{k} = ttm(Xapprox{k},Vbasis{k,r},r);
%             
%         end

%     end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



vanilla = firstnotseparatein;

nsmax  = vanilla.nmax;
kappa = vanilla.kaps;
tf = vanilla.tf;
tau = vanilla.tau;
nn = vanilla.nn + 1;
ode = vanilla.ode;

for k = 1:3
    for r = 1:3
        Uold{k,r} = [];
        FUold{k,r} = [];
    end
    Uvec{k} = [];
    Fvec{k} = [];
end

A = ode.A; X0 = ode.X0;
B = ode.B;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath TENSORTOOLS

nsnaps = nsmax;
h = tf/nsnaps;
tauh = tau/(sqrt(nsnaps)); % This is the tolerance used for the basis truncation


timer = 0;
phasetimer = 1;
i = 1;


%%% Preliminaries %%%



%% The eigenvalue decomposition of the third dimension is required for the T3SYLV solver

for k = 1:3
    
    for r = 3
        
        [VA{k,r},DA{k,r}] = eig(full(A{k,r})');
        ddA{k,r} = diag(DA{k,r});
        VAi{k,r} = inv(VA{k,r});
        
    end
    
end

%%

fullint = tic; % Initiate timers
svdtime = 0;
fulltime = 0;

Lfirst{1} = eye(nn); Lfirst{2} = eye(nn); Lfirst{3} = eye(nn);


%% Start the procedure

timer = 0;

% Form the matrix L used for the quick solving of th LYAP eq.
fprintf('\n\n\n OFFLINE PHASE \n')
fprintf('\n\nSnapshot Capture and Basis construction...\n\n')
fprintf('0------------------tf\n')
 
while timer < tf
    
    if rem(i,nsmax/20) == 0
        fprintf('.')
    end
    
    if phasetimer < 2 %% This includes the starting tensors into the space
        
        for k = 1:3
            Xnew{k} = X0{k};
            X3{k} = double(tenmat(Xnew{k},3,'t'));
        end
        
        
        for k = 1:3
            
            for r = 1:3
                Fnew{k,r} = ttm(Xnew{k},B{r},r).*Xnew{r};
            end
            
            Fnew{k} = Fnew{k,1}+Fnew{k,2}+Fnew{k,3};
        end
        
        phasetimer = phasetimer+1;
        
    else
        
        %%%%%% Here we integrate the function to determine the next U(t_i) %%%%%%%
        
        
        
        for k = 1:3   % fOR EACH EQUATION WE SOLVE THE LINEAR SYTEM WITH T3SYLV
            
            rhs = Xnew{k} + h*Fnew{k};
            rhs = double(tenmat(rhs,3,'t'));
            rhstrans = rhs*VA{k,3};
            
            for l = 1:nn % Solve a sequence of nn sylvester equations to determine the solution
                
                RHS = reshape(rhstrans(:,l),nn,nn);
                
                A11 = (1-h*ddA{k,3}(l))*eye(size(A{k,1})) - h*A{k,1};
                A21 = -h*A{k,2}';
                YY = lyap(A11,A21,-RHS);
                
                
                XX(:,l) = YY(:);
                
            end
            
            X3{k} = XX*VAi{k,3};
            
            Xnew{k} = tensor(X3{k},[nn,nn,nn]);
            
        end
        
        
        %%% Update the nonlinear terms %%%
        
        for k = 1:3
            
            for r = 1:3
                Fnew{k,r} = ttm(Xnew{k},B{r},r).*Xnew{r};
            end
 
            Fnew{k} = Fnew{k,1}+Fnew{k,2}+Fnew{k,3};
        end
        
        
        
    end
    
    
    
    if rem(i,i) == 0 % This can be changed to not include every single snapshot (maybe every second or 5th)
        
        
        svdtimer = tic;
        
        %%% PREPROCESS SNAPSHOT SOLUTIONS %%%%
        
        %%% Each update is done for each equation 'k' and direction 'r'
        
        for k = 1:3
            
            [u{k},perm{k},s{k}] = sthosvd(Xnew{k}, kappa);
            
            for r = 1:3
                N{k,r}=sum(s{k}(r,:)/s{k}(r,1)>tau);
                S{k,r} = s{k}(r,1:N{k,r});
            end
            
            
            %%% PREPROCESS NONLINEAR SNAPSHOT %%%%
            
            
            
            [Fu{k},Fperm{k},Fs{k}] = sthosvd(Fnew{k}, kappa);
            
            
            for r = 1:3
                FN{k,r}=sum(Fs{k}(r,:)/Fs{k}(r,1)>tau);
                FS{k,r} = Fs{k}(r,1:FN{k,r});
                
            end
            
        end
        

        for k = 1:3 % Update the solution bases for all three equations
            
            
            
            for r = 1:3 %Update each of the three spacial directions
                
                CU{k,r} = u{k}.U{r}(:,1:N{k,r});
                
                if isempty(Uold{k,r})
                    Su{k,r} = S{k,r}; Suold{k,r} = Su{k,r};
                    U{k,r} = CU{k,r}; Uold{k,r} = U{k,r}; p{k,r} = size(U{k,r},2); ku{k,r} = p{k,r};
                else
                    
                    Uold{k,r} = Vbasis{k,r};
                    Suold{k,r} = Suold{k,r}(1:p{k,r});
                    
                    if ku{k,r} >= nn
                        U{k,r} = Uold{k,r};
                        Su{k,r} = Suold{k,r};
                    else
                        [U{k,r},su1{k,r},~]=svd([Uold{k,r}*diag(Suold{k,r}),CU{k,r}*diag(S{k,r})],0); strunc{k,r}=diag(su1{k,r});
                        p{k,r}=size([Uold{k,r},CU{k,r}],2)-sum((cumsum(strunc{k,r}(end:-1:1))/sum(strunc{k,r}))<tauh);
                        Suold{k,r} = strunc{k,r};
                        ku{k,r} = size([Uold{k,r},CU{k,r}],2);
                    end
                    
                end
                
                Vbasis{k,r} = U{k,r}(:,1:p{k,r});
                
            end
        end
        
        
        
        for k = 1:3 % Update the nonlinear bases for all three equations
            
            for r = 1:3 %Update each of the three spacial directions
                
                FCU{k,r} = Fu{k}.U{r}(:,1:FN{k,r});
                
                if isempty(FUold{k,r})
                    FSu{k,r} = FS{k,r}; FSuold{k,r} = FSu{k,r};
                    FU{k,r} = FCU{k,r}; FUold{k,r} = FU{k,r}; Fp{k,r} = size(FU{k,r},2); Fku{k,r} = Fp{k,r};
                else
                    
                    FUold{k,r} = FVbasis{k,r};
                    FSuold{k,r} = FSuold{k,r}(1:Fp{k,r});
                    
                    if Fku{k,r} >= nn
                        FU{k,r} = FUold{k,r};
                        FSu{k,r} = FSuold{k,r};
                    else
                        [FU{k,r},Fsu1{k,r},~]=svd([FUold{k,r}*diag(FSuold{k,r}),FCU{k,r}*diag(FS{k,r})],0); Fstrunc{k,r}=diag(Fsu1{k,r});
                        Fp{k,r}=size([FUold{k,r},FCU{k,r}],2)-sum((cumsum(Fstrunc{k,r}(end:-1:1))/sum(Fstrunc{k,r}))<tauh);
                        FSuold{k,r} = Fstrunc{k,r};
                        Fku{k,r} = size([FUold{k,r},FCU{k,r}],2);
                    end
                    
                end
                
                FVbasis{k,r} = FU{k,r}(:,1:Fp{k,r});
                
            end
        end

        
        svdtime = svdtime + toc(svdtimer); % Timer for basis construction
        
    end
    
    timer = timer+h;
    
    i=i+1;

end  % end of while timer


fprintf('\n Elapsed Time For integration and basis construction %d\n',toc(fullint))
fprintf('\n Basis size k1,k2,k3, p1, p2, p3 for eq 1: %d %d %d %d %d %d\n',p{1,1},p{1,2}, p{1,3}, Fp{1,1},Fp{1,2}, Fp{1,3})
fprintf('\n Basis size k1,k2,k3, p1, p2, p3 for eq 2: %d %d %d %d %d %d\n',p{2,1},p{2,2}, p{2,3}, Fp{2,1},Fp{2,2}, Fp{2,3})
fprintf('\n Basis size k1,k2,k3, p1, p2, p3 for eq 3: %d %d %d %d %d %d\n',p{3,1},p{3,2}, p{3,3}, Fp{3,1},Fp{3,2}, Fp{3,3})

fprintf('\n %d seconds spent building the bases  \n',svdtime)




%% Integration of the reduced model %%


% Form the DEIM interolation and oblique projector

clear Xold; clear Xnew; clear Fold; clear Fnew;
i = 1;
krontime = 0;

for k = 1:3
    
    
    Y0{k} = X0{k};
    for r = 1:3
        
        Y0{k} = ttm(Y0{k},Vbasis{k,r},r,'t');
        
        [~,~,II{k,r}]=qr(FVbasis{k,r}','vector');  II{k,r}=II{k,r}(1:Fp{k,r})';
        Deimapprox{k,r}=FVbasis{k,r}/FVbasis{k,r}(II{k,r},:);
        
        % Form the ROM by projection
        
        Ak{k,1} = Vbasis{k,1}'*A{k,1}*Vbasis{k,1};
        Ak{k,2} = Vbasis{k,2}'*A{k,2}*Vbasis{k,2};
        Ak{k,3} = Vbasis{k,3}'*A{k,3}*Vbasis{k,3};
        
        
        Deim{k,r} = Vbasis{k,r}'*Deimapprox{k,r};
        
        
    end
    
    
    [VAsmall{k,3},DAsmall{k,3}] = eig(full(Ak{k,3})'); %eigenvalue decomposition of symmetric A
    ddAsmall{k,3} = diag(DAsmall{k,3});
    VAismall{k,3} = inv(VAsmall{k,3});
    
    
    
end

nt =160;
h = tf/nt;
timer = 0;

fprintf('\n\n\n ONLINE PHASE \n')
fprintf('\n\n\n Integrating the reduced model at %d timesteps \n',nt)
fprintf('\n 0--------------------------------------tf \n')

%%% Initiate Full order model %%%


%%% Initiate reduced order model %%%

for q = 1:3
    for k = 1:3
        Ynew{k} = Y0{k};
        Yrowred{k,q} = Ynew{k};
        for r = 1:3
            Yrowred{k,q} = ttm(Yrowred{k,q},Vbasis{k,r}(II{q,r},:),r);
        end
    end
end


%% Evaluate low-dimensional nonlinear term with DEIM

for k = 1:3
    
    Fsmall1{k} = Ynew{k};
    Fsmall2{k} = Ynew{k};
    Fsmall3{k} = Ynew{k};
    
    Fsmall1{k} = ttm(Fsmall1{k}, B{k,1}(II{k,1},:)*Vbasis{k,1},1);
    Fsmall1{k} = ttm(Fsmall1{k}, Vbasis{k,2}(II{k,2},:),2);
    Fsmall1{k} = ttm(Fsmall1{k}, Vbasis{k,3}(II{k,3},:),3);
    Fsmall1{k} = Fsmall1{k}.*Yrowred{1,k};
    
    Fsmall2{k} = ttm(Fsmall2{k}, B{k,2}(II{k,2},:)*Vbasis{k,2},2);
    Fsmall2{k} = ttm(Fsmall2{k}, Vbasis{k,1}(II{k,1},:),1);
    Fsmall2{k} = ttm(Fsmall2{k}, Vbasis{k,3}(II{k,3},:),3);
    Fsmall2{k} = Fsmall2{k}.*Yrowred{2,k};
    
    Fsmall3{k} = ttm(Fsmall3{k}, Vbasis{k,1}(II{k,1},:),1);
    Fsmall3{k} = ttm(Fsmall3{k}, Vbasis{k,2}(II{k,2},:),2);
    Fsmall3{k} = ttm(Fsmall3{k}, B{k,3}(II{k,3},:)*Vbasis{k,3},3);
    
    Fsmall3{k} = Fsmall3{k}.*Yrowred{3,k};
    
    Fnewsmall{k} =  Fsmall1{k} + Fsmall2{k} + Fsmall3{k};
    
end




for k = 1:3
    for r = 1:3
        Fnewsmall{k} = ttm(Fnewsmall{k} ,Deim{k,r},r);
    end
end


timer = timer+h;
i = i+1;


romtime = 0;
romtime2 = 0;

while timer < tf
    
    if rem(i,floor(nt/40)) == 0
        fprintf('.')
    end
    
    romtimer = tic;
    %%% Reduced order model %%%
    
    for k = 1:3
        
        
        rhsmall{k} = Ynew{k} + h*Fnewsmall{k};
        
        
        romtimer2=tic;
        rhsmall{k} = double(tenmat(rhsmall{k},3,'t'));
        rhstransmall{k} = rhsmall{k}*VAsmall{k,3};
        
        for l = 1:p{k,3} % Solve a sequence of nn sylvester equations to determine the solution

            
            RHSmall{k} = reshape(rhstransmall{k}(:,l),p{k,1},p{k,2});
            
            A11small{k} = (1-h*ddAsmall{k,3}(l))*eye(size(Ak{k,1})) - h*Ak{k,1};
            A21small{k} = -h*Ak{k,2}';
            YYsmall{k} = lyap(A11small{k},A21small{k},-RHSmall{k});
            XXsmall{k}(:,l) = YYsmall{k}(:);
            
        end
        
        XXsmall{k} = XXsmall{k}*VAismall{k,3};
        
        Ynew{k} = tensor(XXsmall{k},[p{k,1},p{k,2},p{k,3}]);
        romtime2 = romtime2+toc(romtimer2);
        
    end
    
    for q = 1:3
        for k = 1:3
            Yrowred{k,q} = Ynew{k};
            for r = 1:3
                Yrowred{k,q} = ttm(Yrowred{k,q},Vbasis{k,r}(II{q,r},:),r);
            end
        end
    end
    
    
    %%% Evaluate low-dimensional nonlinear term
    
    for k = 1:3
        
        Fsmall1{k} = Ynew{k};
        Fsmall2{k} = Ynew{k};
        Fsmall3{k} = Ynew{k};
        
        Fsmall1{k} = ttm(Fsmall1{k}, B{k,1}(II{k,1},:)*Vbasis{k,1},1);
        Fsmall1{k} = ttm(Fsmall1{k}, Vbasis{k,2}(II{k,2},:),2);
        Fsmall1{k} = ttm(Fsmall1{k}, Vbasis{k,3}(II{k,3},:),3);
        Fsmall1{k} = Fsmall1{k}.*Yrowred{1,k};
        
        Fsmall2{k} = ttm(Fsmall2{k}, B{k,2}(II{k,2},:)*Vbasis{k,2},2);
        Fsmall2{k} = ttm(Fsmall2{k}, Vbasis{k,1}(II{k,1},:),1);
        Fsmall2{k} = ttm(Fsmall2{k}, Vbasis{k,3}(II{k,3},:),3);
        Fsmall2{k} = Fsmall2{k}.*Yrowred{2,k};
        
        Fsmall3{k} = ttm(Fsmall3{k}, Vbasis{k,1}(II{k,1},:),1);
        Fsmall3{k} = ttm(Fsmall3{k}, Vbasis{k,2}(II{k,2},:),2);
        Fsmall3{k} = ttm(Fsmall3{k}, B{k,3}(II{k,3},:)*Vbasis{k,3},3);
        
        Fsmall3{k} = Fsmall3{k}.*Yrowred{3,k};
        
        Fnewsmall{k} =  Fsmall1{k} + Fsmall2{k} + Fsmall3{k};
        
    end
    
    
    
    
    for k = 1:3
        for r = 1:3
            Fnewsmall{k} = ttm(Fnewsmall{k} ,Deim{k,r},r);
        end
    end

    romtime = romtime+toc(romtimer);
    

    timer = timer+h;
    i = i+1;
     
end

fprintf('\n')

fprintf('\n %d seconds spent building the bases \n',svdtime)
fprintf('\n ROM integrated at %d timesteps in %d seconds\n',nt,romtime)
fprintf('Done.\n')


end
