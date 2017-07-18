function []=theory_ser()

SNR=-10:2:10;
M=2;
k=log2(M);
nsamp=32;

EbNo=SNR-10*log10(k)+10*log10(nsamp/2);
EbNo=10.^((EbNo)./10);%�ֱ�ֵת��Ϊ��ֵ
y_2ask=0.5*(erfc(sqrt(EbNo/2)));%2ASK�ź���ɽ�����������ʼ���
y_bpsk = berawgn(SNR-10*log10(k)+10*log10(nsamp/2),'psk',M,'nondiff');
y_2fsk = berawgn(SNR-10*log10(k)+10*log10(nsamp/2),'fsk',M,'noncoherent');

semilogy(SNR,y_2ask,SNR,y_bpsk,SNR,y_2fsk,'-b');
legend('theoritical SER of 2ASK', 'theoritical SER of BPSK', 'theoritical SER of 2FSK');
xlabel('�����(dB)')
ylabel('������');
grid on;
title('2ASK��2FSK��BPSK�ź���ɽ��ʱ������������ʵĹ�ϵ');