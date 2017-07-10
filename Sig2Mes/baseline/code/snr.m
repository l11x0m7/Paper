function snr_value=snr(signal, mix_signal)
% signal是原始信号
% mix_signal是加了噪声后的信号
% snr_valur是这个信号的信噪比(dB)


sigPower = sum(abs(signal(:)).^2)/length(signal(:))
noisePower = sum(abs(mix_signal(:)-signal(:)).^2)/length(mix_signal(:));

snr_value_db = 10*log10(sigPower) - 10*log10(noisePower);
snr_value_linear = sigPower / noisePower;
snr_value = [snr_value_linear, snr_value_db];
