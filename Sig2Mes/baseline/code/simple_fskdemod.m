function [res] = simple_fskdemod(y, M, freq_sep, Nsamp, Fs, Fc)
% simple FSK demodulator
% now for 2FSK
% y: input modulation signal sequences
% M: modulation order
% freq_sep: frequency separation
% nSamp: number of samples per symbol
% Fs: base signal frequency
% Fc: carrier wave frequency
f1 = Fc - freq_sep;
f2 = Fc + freq_sep;
t_sin = 1/Fs/Nsamp*(0:Nsamp-1);
carrier1 = cos(2*pi*f1*t_sin);
carrier2 = cos(2*pi*f2*t_sin);

[m, n] = size(y);
res = zeros(m,n/Nsamp);
for symbol = 1:1:m
    for i = 1:Nsamp:n
        z1 = abs(sum(y(symbol,i:i+Nsamp-1).*carrier1) / Fc);
        z2 = abs(sum(y(symbol,i:i+Nsamp-1).*carrier2) / Fc);
%         plot(y(1,i:i+nSamp-1));
%         hold on;
        if z1 >= z2
            res(symbol, (i+Nsamp-1)/Nsamp)=1;
        elseif z1 < z2
            res(symbol, (i+Nsamp-1)/Nsamp)=0;
        end
    end
end