function write_obj_aligned_shapes(ds, ga)
% WRITE_OBJ_ALIGNED_SHAPES - Save aligned surfaces in .obj format
% This function saves a collection of .obj files representing aligned surface 
% meshes.
%
% Syntax: write_obj_aligned_shapes(ds, ga)
%
% Inputs:
%	ds - Structure array of surface shape data (see clusterMapLowRes.m)
%   ga - Structure array of alignment data (see clusterMapLowRes.m, globalize.m)
% 			   	
% Outputs:
% 	None
%
% Toolbox required: ToolBoxGraph

% Author: Julie Winchester, Ph.D.
% Department of Evolutionary Anthropology, Duke University
% Email: jmw110@duke.edu
% Website: http://www.apotropa.com
% January 5, 2017

center = @(X) X-repmat(mean(X,2),1,size(X,2));
scale  = @(X) norm(center(X),'fro') ;

centeredV = cellfun(@(X) center(X.origV), ds.shape, 'UniformOutput', false);
vertexArray = cellfun(@(V,R) R * (V/scale(V)), centeredV, ga.R, 'UniformOutput', false);

if ds.refAlign == 1
	coordR = coord_axis_rotation(vertexArray);
	vertexArray = cellfun(@(X) coordR * X, vertexArray, 'UniformOutput', false); 
end

for ii = 1:ds.n
	write_obj(fullfile(ds.msc.mesh_aligned_dir, [ds.names{ii}, '_aligned.obj']), vertexArray{ii}, ds.shape{ii}.origF);
end