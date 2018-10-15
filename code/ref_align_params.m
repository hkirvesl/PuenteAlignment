function [baseIdx, refAlign] = ref_align_params(align_to, filenames)
% REF_ALIGN_PARAMS - Returns parameters for reference alignment
% Returns 1) the index of the surface mesh to which all other meshes should be 
% aligned (i.e., the mesh for which the transformation matrix is identity), and 
% 2) whether a semi-standardized coordinate axis reference alignment based on 
% principal axes of shape variation should be applied (see 
% coord_axis_rotation.m). 
%
% Syntax: [baseIdx, refAlign] = ref_align_params(align_to, filenames)
%
% Inputs:
%	align_to - User-supplied string, should be 'auto', '', or a filename
%   filenames - Cell array of strings of surface mesh filenames
% 			   	
% Outputs:
% 	baseIdx - Integer index of mesh to which all other meshes should be aligned
%	refAlign - Boolean, whether coordinate axis alignment is to be used

% Author: Julie Winchester, Ph.D.
% Department of Evolutionary Anthropology, Duke University
% Email: jmw110@duke.edu
% Website: http://www.apotropa.com
% January 6, 2017

if ~strcmp(align_to, 'auto') && ~strcmp(align_to, '')
	if any(strcmp(filenames, align_to))
    	disp(['Aligning all meshes to orientation of ' align_to '...']);
    	baseIdx = find(strcmp(filenames, align_to));
    	refAlign = 0;
    else
    	error('Error. Parameter align_to in jadd_path.m not set to ''auto'', '''', or a valid filename.');
    end
else
	disp('Aligning all meshes to semi-standardized reference alignment...')
    baseIdx = 1;
    refAlign = 1;
end