function write_dataset_shard(path,samples,metadata,layout,cfg)
% HDF5 writes are atomic at the shard boundary; completed shards are immutable.
assert(~isfile(path),'Refusing to overwrite an existing dataset shard.');
temp = [path '.partial'];
if isfile(temp), delete(temp); end
put_complex(temp,'y_eq',samples.y);
put_complex(temp,'H_eq',samples.H);
put_complex(temp,'x',samples.x);
put_complex(temp,'ce_coeff',samples.coeff);
put(temp,'class_id',samples.labels);
map = uint8([0 0;0 1;1 1;1 0]);
bits = reshape(map(double(samples.labels(:))+1,:).',2,size(samples.labels,1),[]);
put(temp,'bits',bits);
put(temp,'fit_residual',single(samples.fit_residual));
names = fieldnames(metadata);
for i=1:numel(names), put(temp,names{i},metadata.(names{i})); end
put(temp,'data_bins',int32(layout.data_bins));
put(temp,'pilot_bins',int32(layout.pilot_bins));
h5writeatt(temp,'/','schema_version','1.1');
h5writeatt(temp,'/','matrix_axes_in_h5py','sample,column,row');
h5writeatt(temp,'/','vector_axes_in_h5py','sample,subcarrier');
h5writeatt(temp,'/','snr_definition','received communication-band payload signal/noise before compensation');
h5writeatt(temp,'/','noise_information','simulator-measured AWGN after the frozen receive frontend; excludes CSI error');
h5writeatt(temp,'/','config_json',jsonencode(cfg));
h5writeatt(temp,'/','doppler_source','bookend identical up-LFM correlation peak spacing; known_start unchanged');
movefile(temp,path);
end

function put_complex(path,name,a)
put(path,[name '_real'],single(real(a)));
put(path,[name '_imag'],single(imag(a)));
end

function put(path,name,a)
sz = size(a);
chunk = sz; chunk(end) = min(chunk(end),8);
h5create(path,['/' name],sz,'Datatype',class(a),'ChunkSize',chunk,'Deflate',4);
h5write(path,['/' name],a);
end
