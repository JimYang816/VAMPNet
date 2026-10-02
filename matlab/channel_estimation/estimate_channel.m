function [H,coeff,fit_residual]=estimate_channel(Y,cfg,layout)
% Two regularized training fits; pilot increments retain the training prior.
% Unknown data ICI contaminates pilots; no transmitted data or truth is read.
N=cfg.fft_size;L=cfg.ce_delay_taps;D=numel(cfg.data_symbols);
training=ifft(Y(:,cfg.training_symbols),[],1)*sqrt(N);
c=layout.training_operator*training;
fitted=layout.training_design*c;
errors=sum(abs(training-fitted).^2,1)./max(sum(abs(training).^2,1),eps);
H=complex(zeros(N,N,D));coeff=complex(zeros(L,5,D));fit_residual=zeros(1,D);
center=(N-1)/2;block=N+cfg.cp_samples;
spacing=diff(cfg.training_symbols)*block;
for j=1:D
    m=cfg.data_symbols(j);
    a0=c(1:L,1).';a1=c(1:L,2).';
    b0=c(L+1:end,1).'/N;b1=c(L+1:end,2).'/N;
    u=((m-cfg.training_symbols(1))*block+(0:N-1).'-center)/spacing;
    omega=zeros(1,L);
    if cfg.phase_aware_interpolation
        % Recover phase turns between known slots from locally measured slopes.
        % The energy floor prevents arbitrary unwraps on near-zero taps.
        energy=abs(a0).^2+abs(a1).^2;
        floor_energy=max(max(energy)*1e-3,eps);
        rate=imag(b0.*conj(a0)+b1.*conj(a1))./max(energy,floor_energy);
        phase=angle(a1.*conj(a0));
        omega=(phase+2*pi*round((rate*spacing-phase)/(2*pi)))/spacing;
        omega(energy<floor_energy)=0;
    end
    right=a1.*exp(-1i*omega*spacing);
    slope0=b0-1i*omega.*a0;
    slope1=(b1-1i*omega.*a1).*exp(-1i*omega*spacing);
    h=((2*u.^3-3*u.^2+1)*a0+(u.^3-2*u.^2+u)*(spacing*slope0)+ ...
        (-2*u.^3+3*u.^2)*right+(u.^3-u.^2)*(spacing*slope1)).*exp(1i*(u*spacing)*omega);
    if m>cfg.training_symbols(2)
        % Reuse the second fit as each symbol's prior, with its local slope.
        % Never project that slope across multiple data-symbol centers.
        a0=a1;b0=b1;
        h=ones(N,1)*a1+((0:N-1).'-center)*b1;
        fit_residual(j)=errors(2);
    else
        fit_residual(j)=mean(errors);
    end
    if cfg.pilot_anchor
        prior=tvfir_matrix(h,layout.delay_fourier);
        pp=prior(layout.pilot_idx,layout.pilot_idx);
        obs=(Y(layout.pilot_idx,cfg.data_symbols(j))-(pp-diag(diag(pp)))*layout.pilot_values)./layout.pilot_values;
        increment=layout.anchor_operator*(obs-layout.pilot_design*mean(h,1).');
        h=h+increment.';
    end
    H(:,:,j)=tvfir_matrix(h,layout.delay_fourier);
    coeff(:,:,j)=[a0.' b0.' a1.' b1.' mean(h,1).'];
end
end
