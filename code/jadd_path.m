%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% setup parameters in this section 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% "meshesPath" is where the orignal meshes are located

% meshesPath = '~/Dropbox/transmission/data/PNAS-Platyrrhines/meshes/';
% meshesPath = '/gtmp/BoyerLab/trgao10/PNASExtPlaty319/';
% meshesPath = '/gtmp/BoyerLab/trgao10/PNAS_HDM/';
% meshesPath = '/gtmp/BoyerLab/trgao10/test/';
meshesPath = '/home/grad/sshan/Research/code/auto3dgm-matlab-gorgon/off2/0/';

%%%%% "outputPath" stores intermediate files, re-aligned meshes, and
%%%%% morphologika files
% outputPath = '/gtmp/BoyerLab/trgao10/PNASExtPlaty319_GPR_128_128_MST/';
% outputPath = '/gtmp/BoyerLab/trgao10/PNAS_HDM_output/';
% outputPath = '/gtmp/BoyerLab/trgao10/output/';
outputPath = '/home/grad/sshan/Research/code/auto3dgm-matlab-gorgon/output2/0/';

%%%%% set parameters for the algorithm
restart = 1;

iniNumPts = 128;
finNumPts = 1000;
ssType = 'FPS'; %%% 'FPS' | 'GPR' | 'MD'
type = 'MST'; %%% 'MST' | 'SPC' | 'SDP'

use_cluster = 0;
n_jobs = 1; %%% more nodes, more failure (no hadoop!)
allow_reflection = 1; %%% if set to 0, no reflection will be allowed in
                      %%% the alignments
max_iter = 1000; %%% maximum number of iterations for each pairwise alignment
email_notification = 'trgao10@math.duke.edu'; %%% put your email address here if you would like
                         %%% to get notified when a parallel job finishes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% NO NEED TO MODIFY ANYTHING OTHER THAN THIS FILE!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
codePath= [fileparts(pwd) filesep];
path(pathdef);
path(path, genpath([codePath 'software']));
setenv('MOSEKLM_LICENSE_FILE', [codePath 'software/mosek/mosek.lic'])
