function [] = modulate_generator(Snr, Nsample)
%MATLAB生成三种调制信号，2FSK、2ASK、QPSK，叠加噪声
%后将调制的信号插值显示
% clear;
%---------------------注意
%2
%
%---------------------注意
%---------------------参数定义
% Snr = 5; %dB
% Nsample = 20000; %样本数
Nlength = 32; %每个样本的序列长度
%Npoint = 10000; %生成调制信号的取样点个数

%---------------------细节参数定义
% @@@@@2FSK
Freq_base_2fsk = 8;
Q_2FSK = 2; % 载波频率比
Freq_fsk = Freq_base_2fsk / 2; % 频率间隔 (Hz)，保证正交
Nsamp_fsk = 30; % 每个符号的采样点个数
Fs = 128;  % 采样率（Hz）
Fc = Q_2FSK * Freq_base_2fsk; % 载波频率


% @@@@@2ASK
Freq_base = 8; % 2ASK基频频率（Hz）
Nsamp_ask = 30; %每个符号的采样点个数
Q = 2; %载波频率比
% @@@@@QPSK
Freq_base_qpsk  = 8; % QPSK基频频率（Hz）
Q_QPSK = 2; % 载波频率比
Nsamp_qpsk = 30; %每个符号的采样点个数

%---------------------随机信号生成
Sdata_fsk = randi([0 1], 1 , Nlength * Nsample);  %二进制数据，提供给2FSK调制
Sdata_ask = randi([0 1], 1 , Nlength * Nsample);  %二进制数据，提供给2ASK调制
Sdata_q = randi([0 3], 1 , Nlength * Nsample); %四进制数据，提供给QPSK调制

%---------------------将数据映射为2FSK\2ASK\QPSK符号
% @@@@@@ 2FSK
tlen = Nlength / Freq_base_2fsk; % 秒
base_fsk_data = fskmod(Sdata_fsk',2,Freq_fsk,Nsamp_fsk,Fs);
base_fsk_data = base_fsk_data';
t = (0:tlen/Nlength/Nsamp_fsk:tlen-tlen/Nlength/Nsamp_fsk);
ccos = cos(2*pi*Fc.*t);
csin = sin(2*pi*Fc.*t);
Fsk_data = zeros(Nsample * Nlength * Nsamp_qpsk, 1);
for index = 0 : Nsample - 1
    mid = real(base_fsk_data(index*Nlength*Nsamp_fsk+1:(index+1)*Nlength*Nsamp_fsk)).*ccos-imag(base_fsk_data(index*Nlength*Nsamp_fsk+1:(index+1)*Nlength*Nsamp_fsk)).*csin;
    Fsk_data(index*Nlength*Nsamp_fsk+1:(index+1)*Nlength*Nsamp_fsk) = mid;
end;


%h = dsp.SpectrumAnalyzer('SampleRate',Fs);
%step(h,Fsk_data);
% @@@@@@ 2ASK
t_sin = 1/Freq_base/Nsamp_ask*(0:Nsamp_ask-1);
carry_wave = sin(2*pi*Freq_base*t_sin*Q);
Ask_data = kron(Sdata_ask, carry_wave);
% @@@@@@ QPSK
Qpsk_data = zeros(1,Nsample * Nlength * Nsamp_qpsk);
for index = 0 : Nlength * Nsample * Nsamp_qpsk -1
    Qpsk_data(index+1) = sin(2*pi*Q_QPSK*index/Nsamp_qpsk + Sdata_q(fix(index/Nsamp_qpsk)+1)*pi/2);
end


%--------------------叠加高斯白噪声
Fsk_data_noise = awgn(Fsk_data,Snr,'measured');
Ask_data_noise = awgn(Ask_data,Snr,'measured');
Qpsk_data_noise = awgn(Qpsk_data,Snr,'measured');

Fsk_data_waveform = reshape(Fsk_data_noise,length(Fsk_data_noise)/Nsample, Nsample)';
Ask_data_waveform = reshape(Ask_data_noise, length(Ask_data_noise)/Nsample, Nsample)';
Qpsk_data_waveform = reshape(Qpsk_data_noise, length(Qpsk_data_noise)/Nsample, Nsample)';
Sdata_fsk = reshape(Sdata_fsk, Nlength, Nsample)';
Sdata_ask = reshape(Sdata_ask, Nlength, Nsample)';
Sdata_q = reshape(Sdata_q, Nlength, Nsample)';

%---------------------保存成txt
[~, c_fsk] = size(Sdata_fsk);
[r,c]=size(Fsk_data_waveform);
fid=fopen(sprintf('c:/Users/sky/Desktop/2fsk_%s_%s.txt', string(Snr), string(Nsample)),'w');
for i=1:r
    for j=1:c
        if j ~= c
            fprintf(fid,'%5f,',Fsk_data_waveform(i,j));
        else
            fprintf(fid,'%5f\t',Fsk_data_waveform(i,j));
        end
    end
    for j=1:c_fsk
        if j ~= c_fsk
            fprintf(fid,'%d,',Sdata_fsk(i,j));
        else
            fprintf(fid,'%d\n',Sdata_fsk(i,j));
        end
    end
end
fclose(fid);

[r,c]=size(Ask_data_waveform);
[~, c_ask] = size(Sdata_ask);
fid=fopen(sprintf('c:/Users/sky/Desktop/2ask_%s_%s.txt', string(Snr), string(Nsample)),'w');
for i=1:r
    for j=1:c
        if j ~= c
            fprintf(fid,'%5f,',Ask_data_waveform(i,j));
        else
            fprintf(fid,'%5f\t',Ask_data_waveform(i,j));
        end
    end
    for j=1:c_ask
        if j ~= c_ask
            fprintf(fid,'%d,',Sdata_ask(i,j));
        else
            fprintf(fid,'%d\n',Sdata_ask(i,j));
        end
    end
end
fclose(fid);

[r,c]=size(Qpsk_data_waveform);
[~, c_sq] = size(Sdata_q);
fid=fopen(sprintf('c:/Users/sky/Desktop/qpsk_%s_%s.txt', string(Snr), string(Nsample)), 'w');
for i=1:r
    for j=1:c
        if j ~= c
            fprintf(fid,'%5f,',Qpsk_data_waveform(i,j));
        else
            fprintf(fid,'%5f\t',Qpsk_data_waveform(i,j));
        end
    end
    for j=1:c_sq
        if j ~= c_sq
            fprintf(fid,'%d,',Sdata_q(i,j));
        else
            fprintf(fid,'%d\n',Sdata_q(i,j));
        end
    end
end
fclose(fid);


%---------------------波形抽取显示
% lim = Nsamp_fsk * Nlength;
% t = 1:Nlength;
% subplot(6,1,1);
% plot(Fsk_data_waveform(1,:));
% xlim([0,lim]);
% subplot(6,1,2);
% scatter(t, Sdata_fsk(1,:));
% xlim([1,32]);
% subplot(6,1,3);
% plot(Ask_data_waveform(1,:));
% xlim([0,lim]);
% subplot(6,1,4);
% scatter(t, Sdata_ask(1,:));
% xlim([1,32]);
% subplot(6,1,5);
% plot(Qpsk_data_waveform(1,:));
% xlim([0,lim]);
% subplot(6,1,6);
% scatter(t, Sdata_q(1,:));
% xlim([1,32]);
