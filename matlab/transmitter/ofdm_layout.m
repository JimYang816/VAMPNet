function layout=ofdm_layout(cfg)
% Signed FFT order includes DC; guards occupy both physical band edges.
N=cfg.fft_size;L=cfg.ce_delay_taps;G=cfg.guard_bins_per_edge;
assert(mod(N,2)==0 && G>=0 && 2*G<N);
assert(cfg.cp_samples<N && L-1<=cfg.cp_samples,'Delay model exceeds CP.');
assert(cfg.ce_ridge>0 && cfg.pilot_ridge>0,'Underdetermined CE requires regularization.');
layout.active_bins=(-N/2+G:N/2-G-1).';
layout.guard_bins=setdiff((-N/2:N/2-1).',layout.active_bins);
layout.pilot_bins=(cfg.pilot_first_bin:cfg.pilot_spacing:cfg.pilot_last_bin).';
assert(all(ismember(layout.pilot_bins,layout.active_bins)));
layout.data_bins=setdiff(layout.active_bins,layout.pilot_bins);
layout.active_idx=mod(layout.active_bins,N)+1;
layout.pilot_idx=mod(layout.pilot_bins,N)+1;
layout.data_idx=mod(layout.data_bins,N)+1;
layout.pilot_values=exp(1i*pi/2*(0:numel(layout.pilot_bins)-1).');
saved=rng;cleanup=onCleanup(@()rng(saved));rng(cfg.training_seed,'twister');
layout.training_X=zeros(N,1);
layout.training_X(layout.active_idx)=exp(1i*pi/2*randi([0 3],numel(layout.active_idx),1));
x=ifft(layout.training_X)*sqrt(N);
layout.delay_fourier=exp(-2i*pi*(0:N-1).'*(0:L-1)/N);
A=zeros(N,L);
for l=0:L-1,A(:,l+1)=circshift(x,l);end
% h(n,l)=a(l)+b(l)*(n-center)/N on each known training symbol.
u=((0:N-1).'-(N-1)/2)/N;
B=[A u.*A];
% Ridge also covers ill-conditioned training designs.
layout.condition_number=cond(B);
lambda=cfg.ce_ridge*real(trace(B'*B))/size(B,2);
layout.training_operator=(B'*B+lambda*eye(2*L))\B';
layout.training_design=B;
P=layout.delay_fourier(layout.pilot_idx,:);
% Correct the prior in the observable subspace. 64 pilots do not identify
% 193 independent taps; regularized increments retain the prior nullspace.
lambda=cfg.pilot_ridge*real(trace(P*P'))/size(P,1);
layout.anchor_operator=P'/(P*P'+lambda*eye(size(P,1)));
layout.pilot_design=P;
assert(isequal(sort([cfg.training_symbols(:);cfg.data_symbols(:)]),(1:cfg.symbols_per_frame).'));
assert(isequal(cfg.training_symbols(:).',[1 6]) && ...
    isequal(cfg.data_symbols(:).',[2:5 7:10]),'Expected two training slots with two four-data groups.');
end
