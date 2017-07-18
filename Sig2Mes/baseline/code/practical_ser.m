function []=practical_ser()

SNR=-10:2:10;
M=2;
k=log2(M);
Nsamp=32;
Nsample=4000;
Fs=8;
Fc=16;
freq_sep=4;

y_2ask = zeros(1, length(SNR));
y_2fsk = zeros(1, length(SNR));
y_bpsk = zeros(1, length(SNR));
for snr = SNR
    [Fsk_data_waveform, Ask_data_waveform, Bpsk_data_waveform, ~, ...
        Sdata_2fsk, Sdata_2ask, Sdata_bpsk, ~] = modulate_generator(snr, Nsample);
    ask_demod_out = simple_askdemod(Ask_data_waveform, M, Fs, Fc, Nsamp);
    bpsk_demod_out = simple_pskdemod(Bpsk_data_waveform, M, Fs, Fc, Nsamp);
    fsk_demod_out = simple_fskdemod(Fsk_data_waveform, M, freq_sep, Nsamp, Fs, Fc);
    y_2fsk(1, snr/2+6) = symerr(Sdata_2fsk, fsk_demod_out) / Nsample / Nsamp;
    y_2ask(1, snr/2+6) = symerr(Sdata_2ask, ask_demod_out) / Nsample / Nsamp;
    y_bpsk(1, snr/2+6) = symerr(Sdata_bpsk, bpsk_demod_out) / Nsample / Nsamp;
end

semilogy(SNR,y_2ask,SNR,y_bpsk,SNR,y_2fsk,'-b');
legend('practical SER of 2ASK', 'practical SER of BPSK', 'practical SER of 2FSK');
xlabel('–≈‘Î±»(dB)');
ylabel('ŒÛ¬Î¬ ');
grid on;
title('average symbol error rate of different digital modulation types(2ASK, 2FSK and BPSK)');