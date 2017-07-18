function [f,mag]=practical_spectrum(inputDate,FS)
%���Ƶ���Ƶ��ͼ
%���룺inputDate�������ݣ�Ҫ�����ݳ���Ϊż����ͳһ��������FS����Ƶ��
%�����fƵ�ʣ�mag���߷�����
y = inputDate-mean(inputDate);     %ͨ���źŶ�����ֱ������������Ҫ��ȥ��ֱ��
% y = detrend(inputDate);          %��������������������detrend
L = length(y);                     %���ݳ���
% NFFT=2^nextpow2(L);              %���Բ��û�2�Ŀ����㷨����ע�ⲹ���ֵ�������
NFFT = L;                          %����FFT�ĵ���
Y = fft(y,NFFT);
f = FS/2*linspace(0,1,NFFT/2+1);   %��ЧƵ��
mag = 2*abs(Y(1:NFFT/2+1))/L;      %��Ч������
plot(f,mag)
end
