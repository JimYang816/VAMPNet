root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'matlab')); setup_paths();
fprintf('MATLAB: %s\n',version);
assert(license('test','Signal_Toolbox'),'Signal Processing Toolbox is required.');
assert(~isempty(which('replayfilter')),'Run scripts/install_watermark.py first.');
files=dir(fullfile(root,'data','external','ncs1','NCS1_*.mat'));
assert(numel(files)==60,'Expected all 60 official NCS1 records.');
cfg=jsondecode(fileread(fullfile(root,'configs','simulation.json')));
layout=ofdm_layout(cfg);
fprintf('NCS1 records: %d; data/pilot carriers: %d/%d; unregularized training condition: %.4g\n', ...
    numel(files),numel(layout.data_idx),numel(layout.pilot_idx),layout.condition_number);
