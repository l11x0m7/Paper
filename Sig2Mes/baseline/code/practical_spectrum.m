function [f,mag]=practical_spectrum(inputDate,FS)
%绘制单边频谱图
%输入：inputDate输入数据，要求数据长度为偶数，统一行向量；FS采样频率
%输出：f频率；mag单边幅度谱
y = inputDate-mean(inputDate);     %通常信号都含有直流分量，所以要先去下直流
% y = detrend(inputDate);          %如果有线性趋势项可以用detrend
L = length(y);                     %数据长度
% NFFT=2^nextpow2(L);              %可以采用基2的快速算法，但注意补零幅值会有误差
NFFT = L;                          %计算FFT的点数
Y = fft(y,NFFT);
f = FS/2*linspace(0,1,NFFT/2+1);   %有效频率
mag = 2*abs(Y(1:NFFT/2+1))/L;      %有效幅度谱
plot(f,mag)
end
