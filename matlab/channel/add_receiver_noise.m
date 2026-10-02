function [noisy,info] = add_receiver_noise(clean,cfg,payload_range,snr_db,seed)
% SNR in cfg.noise_bandwidth_hz; real white noise before the receiver.
sample = clean(payload_range);
analytic = sqrt(2)*sample.*exp(-2i*pi*cfg.carrier_hz*(payload_range(:)-1)/cfg.passband_fs);
bb = resample(analytic,cfg.baseband_fs,cfg.passband_fs);
% Discard resampler edge transients in the power measurement.
trim = min(32,floor(numel(bb)/10));
bb = bb(1+trim:end-trim);
power_bb = mean(abs(bb).^2);
variance_passband = power_bb/10^(snr_db/10)*cfg.passband_fs/(2*cfg.noise_bandwidth_hz);
rng(seed,'twister');
noise = sqrt(variance_passband)*randn(size(clean));
noisy = clean+noise;
noise_segment=noise(payload_range);
noise_analytic=sqrt(2)*noise_segment.*exp(-2i*pi*cfg.carrier_hz*(payload_range(:)-1)/cfg.passband_fs);
noise_inband=resample(noise_analytic,cfg.baseband_fs,cfg.passband_fs);
noise_inband=noise_inband(1+trim:end-trim);
info.measured_snr_db=10*log10(power_bb/mean(abs(noise_inband).^2));
info.signal_power = power_bb;
info.variance_passband = variance_passband;
info.seed = seed;
end
