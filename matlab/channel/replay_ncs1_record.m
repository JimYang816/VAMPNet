function [packets,frames,source] = replay_ncs1_record(root,cfg,layout,record_id,max_frames)
file = fullfile(root,'data','external','ncs1',sprintf('NCS1_%03d.mat',record_id));
assert(isfile(file),'Missing official NCS1 record: %s',file);
ch = load(file);
assert(ch.fc==cfg.carrier_hz,'Carrier does not match NCS1.');
assert(abs(ch.fs_tau/ch.fs_t-size(ch.h,2))<1e-6,'Unsupported official replay timing.');
source = struct('record_id',record_id,'fc',ch.fc,'fs_t',ch.fs_t,'fs_tau',ch.fs_tau, ...
    'V0',ch.V0,'snapshots',size(ch.h,1),'delay_bins',size(ch.h,2));
source.duration_seconds = (size(ch.h,1)-1)/ch.fs_t;
source.delay_seconds = size(ch.h,2)/ch.fs_tau;
assert(source.delay_seconds <= (cfg.ce_delay_taps-1)/cfg.baseband_fs, ...
    'Configured delay coverage does not cover the original NCS1 response.');
first = build_frame(cfg,layout,cfg.seed+1000*record_id+1);
L1 = ceil(source.delay_seconds*cfg.passband_fs);
L2 = ceil(0.002*cfg.passband_fs);
packet_len = numel(first.tx)+L1+2*L2;
count = min(floor(source.duration_seconds*cfg.passband_fs/packet_len),max_frames);
assert(count>0,'Frame is longer than the record.');
frames = cell(1,count);
train = zeros(packet_len*count,1);
for k=1:count
    frames{k} = build_frame(cfg,layout,cfg.seed+1000*record_id+k);
    idx = (k-1)*packet_len+L2+(1:numel(first.tx));
    train(idx) = frames{k}.tx;
end
% The unmodified official engine handles all measured taps and time variation.
received = replayfilter(train,cfg.passband_fs,ch.h,ch.fs_t,ch.fs_tau,ch.fc);
% Restore the measured nominal Doppler exactly as official sfetch does.
[p,q] = rat(1/(1-ch.V0/1500));
packets = cell(1,count);
for k=1:count
    raw = received((k-1)*packet_len+(1:packet_len));
    packets{k}.clean = resample(raw,p,q);
    packets{k}.known_start = 1+L2*p/q;
    packets{k}.source_time = ((k-1)*packet_len+L2)/cfg.passband_fs;
    start = 1+(L2+frames{k}.payload_offset*cfg.passband_fs/cfg.baseband_fs)*p/q;
    length_payload = numel(frames{k}.payload_bb)*cfg.passband_fs/cfg.baseband_fs*p/q;
    packets{k}.payload_range = max(1,ceil(start)):min(numel(packets{k}.clean),floor(start+length_payload-1));
end
end
