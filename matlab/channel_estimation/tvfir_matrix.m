function H=tvfir_matrix(h,delay_fourier)
% Exact FFT operator for an in-symbol time-varying circular FIR (CP model).
% H(k,p)=sum_l FFT_n(h(n,l))[k-p]/N * exp(-j*2*pi*p*l/N).
N=size(h,1);Fh=fft(h,[],1)/N;
response=Fh*delay_fourier.'; % deliberately nonconjugating transpose
offset=mod((0:N-1).'-(0:N-1),N)+1;
columns=repmat(1:N,N,1);
H=response(sub2ind([N N],offset,columns));
end
