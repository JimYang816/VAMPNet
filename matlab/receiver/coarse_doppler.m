function [alpha,score,diagnostic]=coarse_doppler(rx,known_start,cfg,layout,frame) %#ok<INUSD>
% Same up-LFM at both frame ends: common correlation delay cancels in spacing.
% Peaks estimate scale only; known_start remains the simulation timing input.
fs=cfg.baseband_fs;limit=cfg.doppler_search_limit;
delay=(cfg.ce_delay_taps-1)/fs;
% Carrier Doppler shifts an upchirp peak, so retain samples before known_start.
margin=ceil((delay+abs(cfg.carrier_hz*limit)*cfg.lfm_seconds/cfg.lfm_bandwidth_hz+.004)*fs);
count=ceil(numel(frame.bb)/(1-limit))+2*margin;
bb=receive_baseband(rx,known_start-margin*cfg.passband_fs/fs,0,count,cfg);
template=frame.lfm_bb;
correlation=conv(bb,conj(flipud(template)),'valid');
energy=conv(abs(bb).^2,ones(numel(template),1),'valid');
peaks=zeros(1,2);scores=peaks;
for j=1:2
    nominal=frame.lfm_offsets(j);
    lo=max(1,floor(margin+nominal/(1+limit)+1-margin));
    hi=min(numel(correlation),ceil(margin+nominal/(1-limit)+1+margin));
    power=abs(correlation(lo:hi)).^2;
    [~,p]=max(power);index=lo+p-1;fraction=0;
    % Interpolate the COMPLEX bandlimited correlation before taking power.
    % Three-point interpolation at 6 kHz biases a nearly one-sample main lobe.
    radius=16;factor=32;
    if index>radius && index+radius<=numel(correlation)
        fine=resample(correlation(index+(-radius:radius)),factor,1);
        center=radius*factor+1;
        region=center+(-factor:factor);
        [~,q]=max(abs(fine(region)).^2);q=region(q);
        v=abs(fine(q+(-1:1))).^2;
        denominator=v(1)-2*v(2)+v(3);sub=0;
        if denominator<0,sub=max(-.5,min(.5,.5*(v(1)-v(3))/denominator));end
        fraction=(q-center+sub)/factor;
    end
    peaks(j)=index-1+fraction-margin;
    scores(j)=abs(correlation(index))^2/max(energy(index)*sum(abs(template).^2),eps);
end
spacing=diff(peaks);assert(spacing>0,'Invalid bookend LFM peak order.');
alpha=diff(frame.lfm_offsets)/spacing-1;
score=min(scores);
diagnostic=struct('peak_samples',peaks,'nominal_spacing_samples',diff(frame.lfm_offsets), ...
    'observed_spacing_samples',spacing,'known_start',known_start,'source','bookend LFM correlation peak spacing');
end
