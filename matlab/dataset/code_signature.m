function digest=code_signature(root)
% Exact input bytes lock resumed shards to one waveform and receiver revision.
names={'generate_dataset','ofdm_layout','build_frame','coarse_doppler', ...
    'estimate_channel','tvfir_matrix','replay_ncs1_record', ...
    'receive_baseband','ofdm_fft','add_receiver_noise','write_dataset_shard','qpsk_symbols','replayfilter'};
md=java.security.MessageDigest.getInstance('SHA-256');
for j=1:numel(names)
    path=which(names{j});assert(startsWith(path,root),'Unexpected function on path: %s',path);
    fid=fopen(path,'rb');assert(fid>=0);bytes=fread(fid,inf,'*uint8');fclose(fid);
    md.update(typecast(bytes,'int8'));
end
digest=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
