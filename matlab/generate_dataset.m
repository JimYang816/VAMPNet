function generate_dataset(mode,record_ids)
% Physical NCS1 replay with known training slots and dense estimated CSI.
if nargin<1,mode='smoke';end
assert(ismember(mode,{'full','smoke'}));root=setup_paths();
cfg=jsondecode(fileread(fullfile(root,'configs','simulation.json')));
expected=1:60;max_frames=inf;
if strcmp(mode,'smoke'),cfg.dataset_id=[cfg.dataset_id '_smoke'];expected=[1 43 52];max_frames=2;end
if nargin<2,record_ids=expected;end
assert(all(ismember(record_ids,expected)));
layout=ofdm_layout(cfg);out=fullfile(root,'data','datasets',cfg.dataset_id);
if ~isfolder(out),mkdir(out);end
source=jsondecode(fileread(fullfile(root,'data','external','source_manifest.json')));
signature=code_signature(root);
manifest_file=fullfile(out,'manifest.json');
if isfile(manifest_file)
    manifest=jsondecode(fileread(manifest_file));
    assert(isequal(manifest.config,cfg) && strcmp(manifest.archive_sha256,source.archive_sha256),'Configuration/source mismatch.');
    assert(strcmp(manifest.code_signature,signature),'Generator/receiver code changed: regenerate the dataset before resuming.');
else
    manifest=struct('schema_version','1.1','status','generating','config',cfg, ...
        'archive_sha256',source.archive_sha256,'shards',struct([]),'records',struct([]), ...
        'code_signature',signature,'pilot_condition_number',layout.condition_number, ...
        'channel_estimator','ridge training FIR fits; first group phase-aware Hermite; second group local second-training prior; pilot ridge increments', ...
        'csi_semantics','estimated physical time-varying FIR; no truth CSI or data labels used in receiver', ...
        'ce_coeff_semantics','prior left value/slope, prior right value/slope, pilot-corrected mean; second group reuses second training at each symbol center', ...
        'receiver_latency','whole frame for bookend LFM scale; first group also uses second training; offline equalization', ...
        'feature_validation','See separate physical probe report; matrix support alone is not proof of true wideband Doppler.');
end
save_manifest(manifest_file,manifest);
for rid=record_ids
    if rid<=42,split='train';snrs=cfg.train_snr_db;
    elseif rid<=51,split='validation';snrs=cfg.train_snr_db;
    else,split='test';snrs=cfg.test_snr_db(:).';end
    needed=false;
    for snr=snrs
        rel=sprintf('%s/record_%03d_snr_%02d.h5',split,rid,snr);
        if isempty(manifest.shards) || ~any(strcmp({manifest.shards.path},rel)),needed=true;end
    end
    if ~needed,continue;end
    fprintf('physical replay record %03d (%s)\n',rid,split);
    [packets,frames,src]=replay_ncs1_record(root,cfg,layout,rid,max_frames);
    if isempty(manifest.records),manifest.records=src;
    elseif ~ismember(rid,[manifest.records.record_id]),manifest.records(end+1)=src;end
    for snr=snrs
        rel=sprintf('%s/record_%03d_snr_%02d.h5',split,rid,snr);
        if ~isempty(manifest.shards) && any(strcmp({manifest.shards.path},rel)),continue;end
        all_samples=[];metadata=[];
        for k=1:numel(packets)
            packet=packets{k};frame=frames{k};
            noise_seed=cfg.seed+100000*rid+100*k+snr;
            [rx,info]=add_receiver_noise(packet.clean,cfg,packet.payload_range,snr,noise_seed);
            [alpha,score,scale_info]=coarse_doppler(rx,packet.known_start,cfg,layout,frame);
            bb=receive_baseband(rx,packet.known_start,alpha,numel(frame.bb),cfg);
            nb=receive_baseband(rx-packet.clean,packet.known_start,alpha,numel(frame.bb),cfg);
            Y=ofdm_fft(bb,frame,cfg);Yn=ofdm_fft(nb,frame,cfg);Yclean=Y-Yn;
            [H,c,fit]=estimate_channel(Y,cfg,layout);
            [Hclean,~,~]=estimate_channel(Yclean,cfg,layout);
            J=numel(cfg.data_symbols);D=numel(layout.data_idx);
            samples=struct('y',complex(zeros(D,J,'single')),'H',complex(zeros(D,D,J,'single')), ...
                'coeff',single(c),'fit_residual',fit,'x',single(frame.x),'labels',frame.labels);
            md=struct('noise_var',mean(abs(Yn(layout.data_idx,cfg.data_symbols)).^2,1), ...
                'snr_db',repmat(single(snr),1,J),'record_id',repmat(uint16(rid),1,J), ...
                'frame_id',repmat(uint16(k),1,J),'symbol_id',uint16(cfg.data_symbols(:).'), ...
                'alpha',repmat(single(alpha),1,J),'doppler_score',repmat(single(score),1,J), ...
                'known_start',repmat(packet.known_start,1,J), ...
                'lfm_first_peak',repmat(scale_info.peak_samples(1),1,J), ...
                'lfm_last_peak',repmat(scale_info.peak_samples(2),1,J), ...
                'tx_seed',repmat(uint32(frame.seed),1,J),'noise_seed',repmat(uint32(noise_seed),1,J), ...
                'source_time',repmat(packet.source_time,1,J),'signal_power',repmat(info.signal_power,1,J), ...
                'measured_snr_db',repmat(single(info.measured_snr_db),1,J), ...
                'noise_passband_variance',repmat(info.variance_passband,1,J), ...
                'model_residual_ratio',zeros(1,J),'observed_ici_proxy',zeros(1,J), ...
                'estimated_beyond_one_ratio',zeros(1,J),'noise_induced_csi_change_ratio',zeros(1,J));
            delta=mod(layout.data_bins-layout.data_bins.'+cfg.fft_size/2,cfg.fft_size)-cfg.fft_size/2;
            for j=1:J
                m=cfg.data_symbols(j);A=H(:,:,j);Ad=A(layout.data_idx,layout.data_idx);
                samples.H(:,:,j)=single(Ad);
                samples.y(:,j)=single(Y(layout.data_idx,m)-A(layout.data_idx,layout.pilot_idx)*layout.pilot_values);
                xp=zeros(cfg.fft_size,1);xp(layout.data_idx)=frame.x(:,j);xp(layout.pilot_idx)=layout.pilot_values;
                md.model_residual_ratio(j)=sum(abs(Yclean(:,m)-A*xp).^2)/max(sum(abs(Yclean(:,m)).^2),eps);
                power=abs(Ad).^2;total=max(sum(power,'all'),eps);
                md.observed_ici_proxy(j)=sum(power(delta~=0))/total;
                md.estimated_beyond_one_ratio(j)=sum(power(abs(delta)>1))/total;
                Ac=Hclean(layout.data_idx,layout.data_idx,j);
                md.noise_induced_csi_change_ratio(j)=sum(abs(Ad-Ac).^2,'all')/max(sum(abs(Ac).^2,'all'),eps);
            end
            if isempty(all_samples),all_samples=samples;metadata=md;
            else
                all_samples.y=cat(2,all_samples.y,samples.y);all_samples.H=cat(3,all_samples.H,samples.H);
                all_samples.coeff=cat(3,all_samples.coeff,samples.coeff);all_samples.x=cat(2,all_samples.x,samples.x);
                all_samples.labels=cat(2,all_samples.labels,samples.labels);
                all_samples.fit_residual=[all_samples.fit_residual samples.fit_residual];
                names=fieldnames(md);for i=1:numel(names),metadata.(names{i})=[metadata.(names{i}) md.(names{i})];end
            end
        end
        path=fullfile(out,rel);folder=fileparts(path);if ~isfolder(folder),mkdir(folder);end
        assert(~isfile(path),'Unindexed shard exists: inspect before resuming.');
        write_dataset_shard(path,all_samples,metadata,layout,cfg);
        entry=struct('path',rel,'split',split,'record_id',rid,'snr_db',snr,'samples',size(all_samples.y,2));
        if isempty(manifest.shards),manifest.shards=entry;else,manifest.shards(end+1)=entry;end
        save_manifest(manifest_file,manifest);fprintf('  %s: %d samples, %d data carriers\n',rel,entry.samples,D);
    end
end
covered=[];
for rid=expected
    need=1;if rid>=52,need=numel(cfg.test_snr_db);end
    if ~isempty(manifest.shards) && sum([manifest.shards.record_id]==rid)==need,covered(end+1)=rid;end
end
if isequal(covered,expected),manifest.status='complete';end
save_manifest(manifest_file,manifest);fprintf('status: %s\n',manifest.status);
end

function save_manifest(path,value)
tmp=[path '.partial'];fid=fopen(tmp,'w');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));clear cleanup;movefile(tmp,path,'f');
end
