function X = get_subsampled_shape( dir, id, N, ssType ) 
%Read already subsampled file, if it exists
%If it doesnt or it does not have enough points, read original off file, subsample, save the subsampled file, and return subsample


if ischar(N)
    N = str2double(N);
end

sub_off_fn = [ dir 'subsampled' filesep ssType filesep num2str(id,'%.3d') '.off' ];
off_fn     = [ dir 'original' filesep num2str(id,'%.3d') '.off' ];

disp(['Reading ' off_fn '...']);
[V,F] = read_off( off_fn );
X                = [];
disp('DONE');
if strcmpi(ssType, 'fps')
    ind = subsample(V, N, X);
elseif strcmpi(ssType, 'gpr')
    ind = gplmk(V, F, N, X);
elseif strcmpi(ssType, 'md')
    ind = reducepatch(F,V,N);
else
    error('unknown subsample type!');
end
X     = V ( :, ind );
if( ~exist([dir filesep 'subsampled'], 'dir') )
    mkdir([dir filesep 'subsampled']);
end
write_off( sub_off_fn, X, [1 2 3]'); %write_off breaks if there are no faces

end
