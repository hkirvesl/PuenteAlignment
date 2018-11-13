function [GPLmkIdx,ptuq,Lambda,MaxCCoVal] = gplmk_new(V, F, numLmk, seed )
% Subsample N points from V, using Gaussian Process landmarking
% Arguments:
%   seed - 3 x M matrix with points to be respected in V, i.e. points that
%   belong to V and that should be the first M points in the resulting X.
%   E.g. when you had a previously subsampled set and you want to just
%   increase the number of sampled points

% parameters
weights_type = 'PointCloud-Curv' % 'Surf-GaussMean'; % type of weights to be computed {'Surf-GaussMean','PointCloud-Curv'}
lambda = 0.45; % for 'Surf-GaussMean', sets the ratio of Gauss- and Mean-curvatures
bandwidth_ratio = 1/3; % bandwidth as a function of average edge length
BNN = 500; % number of NN to condsider (for sparsifying the kernel)

% prepare mesh (center + normalize to unit surface area)
nV = size(V,2);
Center = mean(V,2);
V = bsxfun(@minus,V,Center);
[Area,TriArea] = compute_surface_area(V',F');
VertArea = compute_F2V(F,V)'*TriArea/3;
V = V * sqrt(1/Area);

% compute weights
switch weights_type
    case 'Surf-GaussMean'
        [~,~,~,~,Cmean,Cgauss] = compute_curvature(V,F);
        Lambda = VertArea.*(lambda*abs(Cgauss)/sum(abs(Cgauss))+(1-lambda)*abs(Cmean)/sum(abs(Cmean)));
    case 'PointCloud-Curv'
        [~,curvature] = findPointNormals(V',10);
        Lambda = VertArea.*curvature/sum(curvature);
    otherwise
        error('Invalid weightsType');
end

% compute bandwidth (as a function of mean edge length)
adj = triangulation2adjacency(F,V);
[I,J] = find(tril(adj));
bandwidth = bandwidth_ratio * mean(sqrt(sum((V(:,I)-V(:,J)).^2)));

% computer full kernel
BNN = min(BNN,nV); % ensure BNN is no larger than nV
[idx, dist] = knnsearch(V',V','K',BNN+1);
fullPhi = sparse(repmat(1:nV,1,BNN+1),idx,exp(-dist.^2/bandwidth),nV,nV);
fullPhi = (fullPhi+fullPhi')/2; % symmetrize

disp('Constructing full kernel......');
tic;
fullMatProd = fullPhi * sparse(1:nV,1:nV,Lambda,nV,nV) * fullPhi;
disp(['full kernel constructed in ' num2str(toc) ' sec.']);

% init
KernelTrace = diag(fullMatProd);
MaxCCoVal = zeros(1,numLmk);

% init seed
if isempty(seed)
    [~,maxIdx] = max(KernelTrace);
    seed  = V(:,maxIdx);
end
ind_seed = knnsearch( V', seed' );
if( norm( seed - V(:,ind_seed) , 'fro' ) > 1e-10 )
    error('Some seed point did not belong to the set of points to subsample from');
end
n_seed = length( ind_seed );
GPLmkIdx = [ reshape( ind_seed, 1, n_seed )  zeros( 1, numLmk - n_seed ) ];

% init invKn
invKn = zeros(numLmk);
invKn(1:n_seed,1:n_seed) = inv(fullMatProd(GPLmkIdx(1:n_seed),GPLmkIdx(1:n_seed)));

% compute GPLmks
cback = 0;
for j=(1+n_seed):numLmk
    for cc=1:cback
        fprintf('\b');
    end
    cback = fprintf('Landmark: %4d\n',j);
    
    if j == 2
        %invKn(1:(j-1),1:(j-1)) = 1/fullMatProd(GPLmkIdx(1),GPLmkIdx(1));
        ptuq = KernelTrace - sum(fullMatProd(:,GPLmkIdx(1:(j-1)))'...
            .*(invKn(1:(j-1),1:(j-1))*fullMatProd(GPLmkIdx(1:(j-1)),:)),1)';
    else
        update();
    end
    
    [MaxCCoVal(j),maxUQIdx] = max(ptuq);
    GPLmkIdx(j) = maxUQIdx;
end
j = j+1;
update();

    % nested function for the reccuring update
    function update()
        p = fullMatProd(GPLmkIdx(1:(j-2)),GPLmkIdx(j-1));
        mu = 1./(fullMatProd(GPLmkIdx(j-1),GPLmkIdx(j-1))-p'*invKn(1:(j-2),1:(j-2))*p);
        invKn(1:(j-2),1:(j-1)) = invKn(1:(j-2),1:(j-2))*[eye(j-2)+mu*(p*p')*invKn(1:(j-2),1:(j-2)),-mu*p];
        invKn(j-1,1:(j-1)) = [invKn(1:(j-2),j-1)',mu];
        productEntity = invKn(1:(j-1),1:(j-1))*fullMatProd(GPLmkIdx(1:(j-1)),:);
        ptuq = KernelTrace - sum(fullMatProd(:,GPLmkIdx(1:(j-1)))'...
            .*productEntity,1)';
    end
end

function F2V = compute_F2V(F,V)
nf = size(F,2);
nv = size(V,2);
I = [F(1,:),F(2,:),F(3,:)];
J = [1:nf,1:nf,1:nf]';
S = ones(length(I),1);
F2V = sparse(J,I',S,nf,nv);
end
