function [res]=simple_pskdemod(y, M, Fs, Fc, Nsamp)
% simple ASK demodulator
% now for 2ASK
t_sin = 1/Fs/Nsamp*(0:Nsamp-1);
carrier = sin(2*pi*Fc*t_sin);
[m, n] = size(y);
res = zeros(m,n/Nsamp);
for symbol = 1:1:m
    for i = 1:Nsamp:n
        z = sum(y(symbol,i:i+Nsamp-1).*carrier) / Fc;
        if z < 0
            res(symbol, (i+Nsamp-1)/Nsamp)=1;
        end
    end
end
