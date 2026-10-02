function root = setup_paths()
% Add only project-owned MATLAB folders and the installed official engine.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'matlab')));
vendor = fullfile(root,'third_party','watermark');
if isfolder(vendor), addpath(genpath(vendor)); end
end
