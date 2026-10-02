function Y = ofdm_fft(bb,frame,cfg)
block = cfg.fft_size+cfg.cp_samples;
payload = bb(frame.payload_offset+(1:block*cfg.symbols_per_frame));
payload = reshape(payload,block,cfg.symbols_per_frame);
Y = fft(payload(cfg.cp_samples+1:end,:),[],1)/sqrt(cfg.fft_size);
end
