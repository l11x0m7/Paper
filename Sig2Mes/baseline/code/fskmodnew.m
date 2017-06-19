function y = fskmod(x,M,freq_sep,nSamp,varargin)
%FSKMOD Frequency shift keying modulation
%   Y = FSKMOD(X,M,FREQ_SEP,NSAMP) outputs the complex envelope of the
%   modulation of the message signal X using frequency shift keying modulation. M
%   is the alphabet size and must be an integer power of two.  The message
%   signal must consist of integers between 0 and M-1.  FREQ_SEP is the desired
%   separation between successive frequencies, in Hz.  NSAMP denotes the number
%   of samples per symbol and must be an integer greater than 1.  For two
%   dimensional signals, the function treats each column as one channel.
%
%   Y = FSKMOD(X,M,FREQ_SEP,NSAMP,FS) specifies the sampling frequency (Hz).
%   The default sampling frequency is 1.
%
%   Y = FSKMOD(X,M,FREQ_SEP,NSAMP,FS,PHASE_CONT) specifies the phase continuity
%   across FSK symbols.  PHASE_CONT can be either 'cont' for continuous phase, 
%   or 'discont' for discontinuous phase.  The default is 'cont'.
%
%   Y = FSKMOD(X,M,FREQ_SEP,NSAMP,Fs,PHASE_CONT,SYMBOL_ORDER) specifies how the
%   function assigns binary words to corresponding integers. If SYMBOL_ORDER is
%   set to 'bin' (default), then the function uses a natural binary-coded
%   ordering. If SYMBOL_ORDER is set to 'gray', then the function uses a
%   Gray-coded ordering.
%
%   See also FSKDEMOD, PSKMOD, QAMMOD, PAMMOD, OQPSKMOD.

%   Copyright 1996-2012 The MathWorks, Inc.


% Error checks -----------------------------------------------------------------
if (nargin < 4)
    error(message('comm:fskmod:numarg1'));
end

if (nargin > 7)
    error(message('comm:fskmod:numarg2'));
end

% Check X
if (~isreal(x) || any(any(ceil(x) ~= x)) || ~isnumeric(x))
    error(message('comm:fskmod:xreal1'));
end

% Check that M is a positive integer
if (~isreal(M) || ~isscalar(M) || M<2 || (ceil(M)~=M) || ~isnumeric(M))
    error(message('comm:fskmod:Mreal'));
end

% Check that M is of the form 2^K
if(~isnumeric(M) || ceil(log2(M)) ~= log2(M))
    error(message('comm:fskmod:Mpow2'));
end

%Check that all X are integers within [0,M-1]
if ((min(min(x)) < 0) || (max(max(x)) > (M-1)))
    error(message('comm:fskmod:xreal2'));
end

% Check that the FREQ_SEP is greater than 0
if( ~isnumeric(freq_sep) || ~isscalar(freq_sep) || freq_sep<=0 )
    error(message('comm:fskmod:freqSep'));
end

% Check that NSAMP is an integer greater than 1
if((~isnumeric(nSamp) || (ceil(nSamp) ~= nSamp)) || (nSamp <= 1))
    error(message('comm:fskmod:nSampPos'));
end

% Check Fs
if (nargin >= 5)
Fs = varargin{1};
    if (isempty(Fs))
        Fs = 1;
    elseif (~isreal(Fs) || ~isscalar(Fs) || ~isnumeric(Fs) || Fs<=0)
        error(message('comm:fskmod:FsReal'));
    end
else
    Fs = 1;
end
samptime = 1/Fs;

% Check that the maximum transmitted frequency does not exceed Fs/2
maxFreq = ((M-1)/2) * freq_sep;
if (maxFreq > Fs/2)
    error(message('comm:fskmod:maxFreq'));
end

% Check if the phase is continuous or discontinuous
if (nargin >= 6)
    phase_type = varargin{2};
    %check the phase_type string
    if ~( strcmpi(phase_type,'cont') ||  strcmpi(phase_type,'discont') )
        error(message('comm:fskmod:phaseCont'));
    end

else
    phase_type = 'cont';
end

if (strcmpi(phase_type, 'cont'))
    phase_cont = 1;
else
    phase_cont = 0;
end

% Check SYMBOL_ORDER
if(nargin >= 4 && nargin <= 6 )    
   Symbol_Ordering = 'bin';         % default
else
    Symbol_Ordering = varargin{3};
    if (~ischar(Symbol_Ordering)) || (~strcmpi(Symbol_Ordering,'GRAY')) && (~strcmpi(Symbol_Ordering,'BIN'))
        error(message('comm:fskmod:SymbolOrder'));    
    end
end

% End of error checks ----------------------------------------------------------


% Assure that X, if one dimensional, has the correct orientation
wid = size(x,1);
if (wid == 1)
    x = x(:);
end

% Gray encode if necessary
if (strcmpi(Symbol_Ordering,'GRAY'))
    [~,gray_map] = bin2gray(x,'fsk',M);   % Gray encode
    [~,index]=ismember(x,gray_map);
     x=index-1;
end

% Obtain the total number of channels
[nRows, nChan] = size(x);

% Initialize the phase increments and the oscillator phase for modulator with 
% discontinuous phase.
phaseIncr = (0:nSamp-1)' * (-(M-1):2:(M-1)) * 2*pi * freq_sep/2 * samptime/nSamp;
% phIncrSym is the incremental phase over one symbol, across all M tones.
phIncrSym = phaseIncr(end,:);
% phIncrSamp is the incremental phase over one sample, across all M tones.
phIncrSamp = phaseIncr(2,:);    % recall that phaseIncr(1,:) = 0
OscPhase = zeros(nChan, M);

% phase = nSamp*# of symbols x # of channels
Phase = zeros(nSamp*nRows, nChan);

% Special case for discontinuous-phase FSK: can use a table look-up for speed
if ( (~phase_cont) && ...
        ( floor(nSamp*freq_sep/2 * samptime) ==  nSamp*freq_sep/2 * samptime ) )
    exp_phaseIncr = exp(1i*phaseIncr);
    y = reshape(exp_phaseIncr(:,x+1),nRows*nSamp,nChan);
else
    for iChan = 1:nChan
        prevPhase = 0;
        for iSym = 1:nRows
            % Get the initial phase for the current symbol
            if (phase_cont)
                ph1 = prevPhase;
            else
                ph1 = OscPhase(iChan, x(iSym,iChan)+1);
            end

            % Compute the phase of the current symbol by summing the initial phase
            % with the per-symbol phase trajectory associated with the given M-ary
            % data element.
            Phase(nSamp*(iSym-1)+1:nSamp*iSym,iChan) = ...
                ph1*ones(nSamp,1) + phaseIncr(:,x(iSym,iChan)+1);

            % Update the oscillator for a modulator with discontinuous phase.
            % Calculate the phase modulo 2*pi so that the phase doesn't grow too
            % large.
            if (~phase_cont)
                OscPhase(iChan,:) = ...
                    rem(OscPhase(iChan,:) + phIncrSym + phIncrSamp, 2*pi);
            end

            % If in continuous mode, the starting phase for the next symbol is the
            % ending phase of the current symbol plus the phase increment over one
            % sample.
            prevPhase = Phase(nSamp*iSym,iChan) + phIncrSamp(x(iSym,iChan)+1);
        end
    end
    y = exp(1i*Phase);
end

% Restore the output signal to the original orientation
if(wid == 1)
    y = y.';
end

% EOF --- fskmod.m

