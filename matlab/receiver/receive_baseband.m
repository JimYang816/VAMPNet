function bb = receive_baseband(rx,known_start,alpha,nsamples,cfg)
% Single packet-wide dilation; filtering uses MATLAB's delay-compensated FIR.
count = round(nsamples*cfg.passband_fs/cfg.baseband_fs);
positions = known_start + (0:count-1).'/(1+alpha);
z = interp1((1:numel(rx)).',rx,positions,'linear',0);
t = (0:count-1).'/cfg.passband_fs;
bb = resample(sqrt(2)*z.*exp(-2i*pi*cfg.carrier_hz*t),cfg.baseband_fs,cfg.passband_fs);
bb = bb(1:nsamples);
end
