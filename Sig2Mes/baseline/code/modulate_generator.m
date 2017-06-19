function [] = modulate_generator(Snr, Nsample)
%MATLAB�������ֵ����źţ�2FSK��2ASK��QPSK����������
%�󽫵��Ƶ��źŲ�ֵ��ʾ
% clear;
%---------------------ע��
%2
%
%---------------------ע��
%---------------------��������
% Snr = 5; %dB
% Nsample = 20000; %������
Nlength = 32; %ÿ�����������г���
%Npoint = 10000; %���ɵ����źŵ�ȡ�������

%---------------------ϸ�ڲ�������
% @@@@@2FSK
Freq_base_2fsk = 8;
Q_2FSK = 2; % �ز�Ƶ�ʱ�
Freq_fsk = Freq_base_2fsk / 2; % Ƶ�ʼ�� (Hz)����֤����
Nsamp_fsk = 30; % ÿ�����ŵĲ��������
Fs = 128;  % �����ʣ�Hz��
Fc = Q_2FSK * Freq_base_2fsk; % �ز�Ƶ��


% @@@@@2ASK
Freq_base = 8; % 2ASK��ƵƵ�ʣ�Hz��
Nsamp_ask = 30; %ÿ�����ŵĲ��������
Q = 2; %�ز�Ƶ�ʱ�
% @@@@@QPSK
Freq_base_qpsk  = 8; % QPSK��ƵƵ�ʣ�Hz��
Q_QPSK = 2; % �ز�Ƶ�ʱ�
Nsamp_qpsk = 30; %ÿ�����ŵĲ��������

%---------------------����ź�����
Sdata_fsk = randi([0 1], 1 , Nlength * Nsample);  %���������ݣ��ṩ��2FSK����
Sdata_ask = randi([0 1], 1 , Nlength * Nsample);  %���������ݣ��ṩ��2ASK����
Sdata_q = randi([0 3], 1 , Nlength * Nsample); %�Ľ������ݣ��ṩ��QPSK����

%---------------------������ӳ��Ϊ2FSK\2ASK\QPSK����
% @@@@@@ 2FSK
tlen = Nlength / Freq_base_2fsk; % ��
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


%--------------------���Ӹ�˹������
Fsk_data_noise = awgn(Fsk_data,Snr,'measured');
Ask_data_noise = awgn(Ask_data,Snr,'measured');
Qpsk_data_noise = awgn(Qpsk_data,Snr,'measured');

Fsk_data_waveform = reshape(Fsk_data_noise,length(Fsk_data_noise)/Nsample, Nsample)';
Ask_data_waveform = reshape(Ask_data_noise, length(Ask_data_noise)/Nsample, Nsample)';
Qpsk_data_waveform = reshape(Qpsk_data_noise, length(Qpsk_data_noise)/Nsample, Nsample)';
Sdata_fsk = reshape(Sdata_fsk, Nlength, Nsample)';
Sdata_ask = reshape(Sdata_ask, Nlength, Nsample)';
Sdata_q = reshape(Sdata_q, Nlength, Nsample)';

%---------------------�����txt
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


%---------------------���γ�ȡ��ʾ
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
