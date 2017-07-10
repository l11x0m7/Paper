function [] = mix_data_generator(Nsample)
%MATLAB����4�ֵ����źţ�2FSK��2ASK��BPSK��QPSK����������
%�󽫵��Ƶ��źŲ�ֵ��ʾ
%�����źŵķ�������һ��(Ƶ�����һ��)���ز�Ƶ��һ��(��ͬһƵ��)
% clear;
%---------------------��������
Snrs = -10:5:20; %dB
% Nsample = 5000; %������
Nlength = 32; %ÿ�����������г���
Nsignals = 3;
totalSample = Nsample * Nsignals * length(Snrs);

fid=fopen(sprintf('c:/Users/sky/Desktop/mix_%s_%s_%s.txt', string(min(Snrs)), string(max(Snrs)), string(totalSample)), 'w');
for Snr = Snrs
    %---------------------ϸ�ڲ�������
    % @@@@@2FSK
    Freq_base_2fsk = 8;
    Q_2FSK = 2; % �ز�Ƶ�ʱ�
    Freq_2fsk = Freq_base_2fsk / 2; % Ƶ�ʼ�� (Hz)����֤����
    Nsamp_2fsk = 32; % ÿ�����ŵĲ��������
    Fs = 128;  % �����ʣ�Hz��
    Fc_2FSK = Q_2FSK * Freq_base_2fsk; % �ز�Ƶ��

    % @@@@@2ASK
    Freq_base_2ask = 8; % 2ASK��ƵƵ�ʣ�Hz��
    Q_2ASK = 2; %�ز�Ƶ�ʱ�
    Nsamp_2ask = 32; %ÿ�����ŵĲ��������

    % @@@@@BPSK
    Freq_base_bpsk = 8; % BPSK��ƵƵ�ʣ�Hz��
    Q_BPSK = 2; % �ز�Ƶ�ʱ�
    Nsamp_bpsk = 32; %ÿ�����ŵĲ��������

    % @@@@@QPSK
    % Freq_base_qpsk  = 8; % QPSK��ƵƵ�ʣ�Hz��
    % Q_QPSK = 2; % �ز�Ƶ�ʱ�
    % Nsamp_qpsk = 32; %ÿ�����ŵĲ��������


    %---------------------����ź�����
    Sdata_2fsk = randi([0 1], 1 , Nlength * Nsample);  %���������ݣ��ṩ��2FSK����
    Sdata_2ask = randi([0 1], 1 , Nlength * Nsample);  %���������ݣ��ṩ��2ASK����
    Sdata_bpsk = randi([0 1], 1 , Nlength * Nsample); %�Ľ������ݣ��ṩ��QPSK����
    % Sdata_qpsk = randi([0 3], 1 , Nlength * Nsample); %�Ľ������ݣ��ṩ��QPSK����


    %---------------------������ӳ��Ϊ2FSK\2ASK\BPSK\QPSK����
    % @@@@@@ 2FSK
    tlen = Nlength / Freq_base_2fsk; % ��
    base_fsk_data = fskmod(Sdata_2fsk',2,Freq_2fsk,Nsamp_2fsk,Fs);
    base_fsk_data = base_fsk_data';
    t = (0:tlen/Nlength/Nsamp_2fsk:tlen-tlen/Nlength/Nsamp_2fsk);
    ccos = cos(2*pi*Fc_2FSK.*t);
    csin = sin(2*pi*Fc_2FSK.*t);
    Fsk_data = zeros(Nsample * Nlength * Nsamp_2fsk, 1);
    for index = 0 : Nsample - 1
        mid = real(base_fsk_data(index*Nlength*Nsamp_2fsk+1:(index+1)*Nlength*Nsamp_2fsk)).*ccos-imag(base_fsk_data(index*Nlength*Nsamp_2fsk+1:(index+1)*Nlength*Nsamp_2fsk)).*csin;
        Fsk_data(index*Nlength*Nsamp_2fsk+1:(index+1)*Nlength*Nsamp_2fsk) = mid;
    end;

    %h = dsp.SpectrumAnalyzer('SampleRate',Fs);
    %step(h,Fsk_data);
    % @@@@@@ 2ASK
    t_sin = 1/Freq_base_2ask/Nsamp_2ask*(0:Nsamp_2ask-1);
    carry_wave = sin(2*pi*Freq_base_2ask*t_sin*Q_2ASK);
    Ask_data = kron(Sdata_2ask, carry_wave);

    % @@@@@@ BPSK
    Bpsk_data = zeros(1,Nsample * Nlength * Nsamp_bpsk);
    for index = 0 : Nlength * Nsample * Nsamp_bpsk -1
        Bpsk_data(index+1) = sin(2*pi*Q_BPSK*index/Nsamp_bpsk + Sdata_bpsk(fix(index/Nsamp_bpsk)+1)*pi);
    end

    % @@@@@@ QPSK
    % Qpsk_data = zeros(1,Nsample * Nlength * Nsamp_qpsk);
    % for index = 0 : Nlength * Nsample * Nsamp_qpsk -1
    %     Qpsk_data(index+1) = sin(pi/4+2*pi*Q_QPSK*index/Nsamp_qpsk + Sdata_qpsk(fix(index/Nsamp_qpsk)+1)*pi/2);
    % end

    Fsk_data_waveform = reshape(Fsk_data, length(Fsk_data)/Nsample, Nsample)';
    Ask_data_waveform = reshape(Ask_data, length(Ask_data)/Nsample, Nsample)'.*sqrt(2);
    Bpsk_data_waveform = reshape(Bpsk_data, length(Bpsk_data)/Nsample, Nsample)';

    % �������ֵ��źŵȹ��ʣ��Ҷ���ÿ���źţ������ԼΪ-3dB
    Mix_3_signal_data_waveform = Fsk_data_waveform + Ask_data_waveform + Bpsk_data_waveform;

    %--------------------�Ի���źŵ��Ӹ�˹������
    if Snr ~= 'none'
        actual_Snr = round(-10 * log10(2 + 3 / (10^(Snr / 10))), 1);
        Mix_data_noise = awgn(Mix_3_signal_data_waveform, Snr, 'measured');
    else
        actual_Snr = -3;
        Mix_data_noise = Mix_3_signal_data_waveform;
    end

    %--------------------��������
    Sdata_2fsk = reshape(Sdata_2fsk, Nlength, Nsample)';
    Sdata_2ask = reshape(Sdata_2ask, Nlength, Nsample)';
    Sdata_bpsk = reshape(Sdata_bpsk, Nlength, Nsample)';
    % Sdata_qpsk = reshape(Sdata_qpsk, Nlength, Nsample)';

    %---------------------�����txt
    [~, c_2fsk] = size(Sdata_2fsk);
    [~, c_2ask] = size(Sdata_2ask);
    [~, c_bpsk] = size(Sdata_bpsk);
    [r,c]=size(Mix_data_noise);
    for i=1:r
        for j=1:c
            if j ~= c
                fprintf(fid,'%.5f,',Mix_data_noise(i,j));
            else
                fprintf(fid,'%.5f\t',Mix_data_noise(i,j));
            end
        end
        for j=1:c_2ask
            if j ~= c_2ask
                fprintf(fid,'%d,',Sdata_2ask(i,j));
            else
                fprintf(fid,'%d\t',Sdata_2ask(i,j));
            end
        end
        fprintf(fid, '%.1f\t', Snr);
        fprintf(fid, '%.1f\t', actual_Snr);
        fprintf(fid, '2ASK\n');
    end

    for i=1:r
        for j=1:c
            if j ~= c
                fprintf(fid,'%.5f,',Mix_data_noise(i,j));
            else
                fprintf(fid,'%.5f\t',Mix_data_noise(i,j));
            end
        end
        for j=1:c_2fsk
            if j ~= c_2fsk
                fprintf(fid,'%d,',Sdata_2fsk(i,j));
            else
                fprintf(fid,'%d\t',Sdata_2fsk(i,j));
            end
        end
        fprintf(fid, '%.1f\t', Snr);
        fprintf(fid, '%.1f\t', actual_Snr);
        fprintf(fid, '2FSK\n');
    end

    for i=1:r
        for j=1:c
            if j ~= c
                fprintf(fid,'%.5f,',Mix_data_noise(i,j));
            else
                fprintf(fid,'%.5f\t',Mix_data_noise(i,j));
            end
        end
        for j=1:c_bpsk
            if j ~= c_bpsk
                fprintf(fid,'%d,',Sdata_bpsk(i,j));
            else
                fprintf(fid,'%d\t',Sdata_bpsk(i,j));
            end
        end
        fprintf(fid, '%.1f\t', Snr);
        fprintf(fid, '%.1f\t', actual_Snr);
        fprintf(fid, 'BPSK\n');
    end

end

fclose(fid);

%---------------------���γ�ȡ��ʾ
% lim = Nsamp_2fsk * Nlength;
% t = 1:Nlength;
% subplot(8,1,1);
% plot(Fsk_data_waveform(1,1:32));
% xlim([0,lim]);
% subplot(8,1,2);
% scatter(t, Sdata_2fsk(1,:));
% xlim([1,32]);
% subplot(8,1,3);
% plot(Ask_data_waveform(1,:));
% xlim([0,lim]);
% subplot(8,1,4);
% scatter(t, Sdata_2ask(1,:));
% xlim([1,32]);
% subplot(8,1,5);
% plot(Bpsk_data_waveform(1,:));
% xlim([0,lim]);
% subplot(8,1,6);
% scatter(t, Sdata_bpsk(1,:));
% xlim([1,32]);
% subplot(8,1,7);
% plot(Qpsk_data_waveform(1,:));
% xlim([0,lim]);
% subplot(8,1,8);
% scatter(t, Sdata_qpsk(1,:));
% xlim([1,32]);

%-----------------��ʾ�źŵ�Ƶ��
% subplot(4,1,1);
% practical_spectrum(Ask_data_waveform(1,:),256);
% xlim([0,140])
% subplot(4,1,2);
% practical_spectrum(Fsk_data_waveform(1,:),256);
% xlim([0,140])
% subplot(4,1,3);
% practical_spectrum(Bpsk_data_waveform(1,:),256);
% xlim([0,140]);
% subplot(4,1,4);
% practical_spectrum(Qpsk_data_waveform(1,:),256);
% xlim([0,140]);