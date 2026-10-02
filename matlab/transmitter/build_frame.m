function frame = build_frame(cfg,layout,seed)
rng(seed,'twister');
N = cfg.fft_size; M = cfg.symbols_per_frame;
frame.labels = uint8(randi([0 3],numel(layout.data_idx),M));
[frame.x,~] = qpsk_symbols(frame.labels);
X = complex(zeros(N,M));
X(layout.data_idx,:) = frame.x;
X(layout.pilot_idx,:) = repmat(layout.pilot_values,1,M);
X(:,cfg.training_symbols) = repmat(layout.training_X,1,numel(cfg.training_symbols));
payload = ifft(X,[],1)*sqrt(N);
payload = [payload(end-cfg.cp_samples+1:end,:);payload];
frame.payload_bb = payload(:);
nlfm = round(cfg.lfm_seconds*cfg.baseband_fs);
t = (0:nlfm-1).'/cfg.baseband_fs;
B = cfg.lfm_bandwidth_hz;
assert(B<=cfg.baseband_fs && strcmp(cfg.lfm_mode,'bookend_identical_up'));
up_chirp = exp(1i*2*pi*(-B/2*t+B/(2*cfg.lfm_seconds)*t.^2));
guard = zeros(round(cfg.guard_seconds*cfg.baseband_fs),1);
frame.lfm_bb = up_chirp;
% Equal expected average power for preamble and payload, no payload AGC.
frame.lfm_bb = frame.lfm_bb*sqrt((numel(layout.data_idx)+numel(layout.pilot_idx))/N);
frame.payload_offset = numel(frame.lfm_bb)+numel(guard);
frame.lfm_offsets = [0 frame.payload_offset+numel(frame.payload_bb)+numel(guard)];
frame.bb = [frame.lfm_bb;guard;frame.payload_bb;guard;frame.lfm_bb];
frame.segment_samples = [nlfm numel(guard) numel(frame.payload_bb) numel(guard) nlfm];
up = resample(frame.bb,cfg.passband_fs,cfg.baseband_fs);
tp = (0:numel(up)-1).'/cfg.passband_fs;
frame.tx = sqrt(2)*real(up.*exp(2i*pi*cfg.carrier_hz*tp));
frame.seed = seed;
% Only unknown data slots are labels; known slots are defined by the layout.
frame.x = frame.x(:,cfg.data_symbols);
frame.labels = frame.labels(:,cfg.data_symbols);
end
