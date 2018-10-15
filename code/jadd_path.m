%<<<<<<< HEAD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% NO NEED TO MODIFY ANYTHING OTHER THAN THIS FILE!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%>>>>>>> d0dddb3294f936c1d498fdc04e0fae58610afc62
%%%%% setup parameters in this section 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% "meshesPath" is where the orignal meshes are located
%<<<<<<< HEAD

% meshesPath = '~/Dropbox/transmission/data/PNAS-Platyrrhines/meshes/';
% meshesPath = '/gtmp/BoyerLab/trgao10/PNASExtPlaty319/';
meshesPath = '~/Desktop/tutkimus/SLICER/testdata/20_Test_Teeth/';
% meshesPath = '/gtmp/BoyerLab/trgao10/test/';

%%%%% "outputPath" stores intermediate files, re-aligned meshes, and
%%%%% morphologika files
% outputPath = '/gtmp/BoyerLab/trgao10/PNASExtPlaty319_GPR_128_128_MST/';
outputPath = '~/Desktop/tutkimus/SLICER/testoutput11/';
% outputPath = '/gtmp/BoyerLab/trgao10/output/';

%%%%% set parameters for the algorithm
restart = 1;

iniNumPts = 128;
finNumPts = 256;
ssType = 'FPS'; %%% 'FPS' | 'GPR'
type = 'MST'; %%% 'MST' | 'SPC' | 'SDP'

use_cluster = 0;
n_jobs = 120; %%% more nodes, more failure (no hadoop!)
allow_reflection = 1; %%% if set to 0, no reflection will be allowed in
                      %%% the alignments
max_iter = 1000; %%% maximum number of iterations for each pairwise alignment
email_notification = ''; %%% put your email address here if you would like
                         %%% to get notified when a parallel job finishes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% NO NEED TO MODIFY ANYTHING OTHER THAN THIS FILE!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======
%meshesPath = '/gtmp/BoyerLab/julie/prime/derived/simp10ksmooth100mlclean_off';

%%%%% "outputPath" stores intermediate files, re-aligned meshes, and
%%%%% morphologika files
%outputPath = '/gtmp/BoyerLab/julie/prime_auto3dgm_result';

%%%%% set parameters for the algorithm
%restart = 1;
%iniNumPts = 200;
%finNumPts = 1000;
allow_reflection = 1;
max_iter = 3000;
%use_cluster = 0;
%n_jobs = 100; %%% more nodes, more failure (no hadoop!)
%email_notification = '';

%%%%% experimental parameters
align_to = 'auto';
do_tangent_pca = 1;
do_procrustes_dist_output = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% do not modify anything beyond this point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%>>>>>>> d0dddb3294f936c1d498fdc04e0fae58610afc62
codePath= [fileparts(pwd) filesep];
path(pathdef);
path(path, genpath(fullfile(codePath, 'software', 'RectangularAssignment')));
path(path, genpath(fullfile(codePath, 'software', 'ToolboxGraph')));

if verLessThan('matlab', '7.9')
	warning(['Auto3dgm dependency Mosek does not support MATLAB versions ' ...
		'older than r2009b. Auto3dgm may crash.']);
	path(path, genpath(fullfile(codePath, 'software', 'mosek', '7', 'toolbox', 'r2009b')));
elseif verLessThan('matlab', '7.14')
	if ismac
		warning(['Auto3dgm dependency Mosek does not support OSX MATLAB ' ...
			'versions older than r2012a. Auto3dgm may crash.']);
    end
	path(path, genpath(fullfile(codePath, 'software', 'mosek', '7', 'toolbox', 'r2009b')));
elseif verLessThan('matlab', '8.1')
	path(path, genpath(fullfile(codePath, 'software', 'mosek', '7', 'toolbox', 'r2012a')));
else
	path(path, genpath(fullfile(codePath, 'software', 'mosek', '7', 'toolbox', 'r2013a')));
end

setenv('MOSEKLM_LICENSE_FILE', [codePath 'software/mosek/mosek.lic'])

