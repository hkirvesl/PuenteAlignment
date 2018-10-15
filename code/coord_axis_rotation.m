function coordR = coord_axis_rotation(vertexArray)
% COORD_AXIS_ROTATION - Return rotation matrix to orient PCAs to coordinate axes
% For a collection of centered scaled aligned surface meshes, derives principal 
% axes of shape variation and calculates rotation matrix to align first, second,
% and third principal axes with Y, X, and Z coordinate axes respectively. 
% Applying rotation matrix to surface meshes orients surfaces so that principal
% axes of shape variation correspond to coordinate axes, providing a 
% semi-standardized orientation in 3D space. 
%
% Syntax: coordR = coord_axis_rotation(vertexArray)
%
% Inputs:
%	vertexArray - Cell array of 3xn surface shape XYZ vertices
% 			   	
% Outputs:
% 	coordR - 3x3 rotation matrix aligning PCAs to coordinate axes

% Author: Julie Winchester, Ph.D.
% Department of Evolutionary Anthropology, Duke University
% Email: jmw110@duke.edu
% Website: http://www.apotropa.com
% January 5, 2017

center = @(X) X-repmat(mean(X,2),1,size(X,2));

vertex = cell2mat(vertexArray);

w = pca(vertex');
pcVector = center(w);
axisVector = center([0, 1, 0; 1, 0, 0; 0, 0, 1]');
covariance = pcVector * axisVector';
[svdU, ~, svdV] = svd(covariance);
coordR = svdV * svdU';

if det(coordR) < 0
	disp('coordR is reflection matrix; changing sign of svdV 3rd column and recalculating...');
	svdV(:,3) = -svdV(:,3);
	coordR = svdV * svdU';
end