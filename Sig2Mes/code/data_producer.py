import numpy as np
import json
from utils import *

def addTwoSignal(signal1, signal2):
    return np.asarray(signal1) + np.asarray(signal2)

def mixSignal(savepath, M, sample_num, *filepaths):
    signals = list()
    messages = list()
    signal_types = list()
    with open(savepath, 'wb') as fw:
        for i, filepath in enumerate(filepaths):
            with open(filepath) as fr:
                cur_signals = list()
                cur_messages = list()
                for row_no, line in enumerate(fr):
                    if sample_num is not None and row_no == sample_num:
                        break
                    signal, message = line.strip().split('\t')
                    signal = map(float, signal.split(','))
                    cur_signals.append(signal)
                    cur_messages.append(message)
                cur_signals = np.asarray(cur_signals)
                signals.append(cur_signals)
                messages.append(cur_messages)
            signal_type = i
            signal_types.append(signal_type)
            print 'File {} parsed'.format(filepath)
        signals = np.sum(signals, axis=0)
        signals = map(lambda row: ','.join(map(str, row)), signals.tolist())
        total_signal = len(signals)
        for signal_type, message in zip(signal_types, messages):
            for i, (signal, mes) in enumerate(zip(signals, message)):
                fw.write('\t'.join(
                    [signal,
                     str(signal_type),
                     mes]) + '\n')
                if (i+1) % 1000 == 0:
                    sys.stdout.write('\rFinish {}/{}'.format(i+1, total_signal))
                    sys.stdout.flush()
    os.system('shuf {} > /tmp/data/tmp_file'.format(savepath))
    os.system('mv /tmp/data/tmp_file {}'.format(savepath))



if __name__ == '__main__':
    mixSignal('../data/10dB/mix_60000.txt', 2, None,
              '../data/10dB/2ask_20000.txt', '../data/10dB/2fsk_20000.txt',
              '../data/10dB/bpsk_20000.txt')

