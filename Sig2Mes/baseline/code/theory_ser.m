function []=theory_ser()

SNR=-10:2:10;
M=2;
k=log2(M);
nsamp=32;

EbNo=SNR-10*log10(k)+10*log10(nsamp/2);
EbNo=10.^((EbNo)./10);%分贝值转化为真值
y_2ask=0.5*(erfc(sqrt(EbNo/2)));%2ASK信号相干解调理论误码率计算
y_bpsk = berawgn(SNR-10*log10(k)+10*log10(nsamp/2),'psk',M,'nondiff');
y_2fsk = berawgn(SNR-10*log10(k)+10*log10(nsamp/2),'fsk',M,'noncoherent');

semilogy(SNR,y_2ask,SNR,y_bpsk,SNR,y_2fsk,'-b');
legend('theoritical SER of 2ASK', 'theoritical SER of BPSK', 'theoritical SER of 2FSK');
xlabel('信噪比(dB)')
ylabel('误码率');
grid on;
title('2ASK、2FSK、BPSK信号相干解调时信噪比与误码率的关系');