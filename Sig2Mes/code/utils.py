import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import re
from collections import defaultdict

def oneHot(val, dim):
    vec = [0] * dim
    vec[int(val)] = 1
    return np.asarray(vec)

def loadSignal(filepath, M, sample_num):
    with open(filepath) as fr:
        signals = list()
        messages = list()
        signal_types = list()
        dbs = list()
        for i, line in enumerate(fr):
            if i == sample_num:
                break
            info = line.strip().split('\t')
            if len(info) == 2:
                signal, message = info[0], info[1]
            else:
                signal, message, st, db = info
            signal = map(float, signal.split(','))
            message = map(float, message.split(','))
            signal = np.asarray(signal)
            signal = signal.reshape(signal.shape[0], 1)
            message = np.asarray(map(lambda val: oneHot(val, M), message))
            signals.append(signal)
            messages.append(message)
            signal_types.append(st)
            dbs.append(db)
        if len(info) == 2:
            return np.asarray(signals), np.asarray(messages)
        else:
            return np.asarray(signals), np.asarray(messages), \
                   np.asarray(signal_types,dtype=np.str), \
                   np.asarray(dbs, dtype=np.str)

def loadMixSignal(filepath, M, mix_type, sample_num):
    with open(filepath) as fr:
        signals = list()
        types = list()
        messages = list()
        for i, line in enumerate(fr):
            if i == sample_num:
                break
            signal, type, message = line.strip().split('\t')
            signal = map(float, signal.split(','))
            type = int(type)
            message = map(float, message.split(','))
            signal = np.asarray(signal)
            signal = signal.reshape(signal.shape[0], 1)
            message = np.asarray(map(lambda val: oneHot(val, M), message))
            type = oneHot(type, mix_type)
            signals.append(signal)
            messages.append(message)
            types.append(type)
        return np.asarray(signals), np.asarray(types), np.asarray(messages)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def save_confusion_matrix(cm, savepath, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True symbol')
    plt.xlabel('Predicted symbol')
    plt.savefig(savepath)
    plt.clf()

def drawSERCurve():
    info_list = """
-10dB 2ask 0.1023515625
-10dB 2fsk 0.134375
-10dB bpsk 0.0371953125
-8dB 2ask 0.0560859375
-8dB 2fsk 0.0713515625
-8dB bpsk 0.0120703125
-6dB 2ask 0.0221953125
-6dB 2fsk 0.026921875
-6dB bpsk 0.0025625
-4dB 2ask 0.0054921875
-4dB 2fsk 0.0065078125
-4dB bpsk 0.000203125
-2dB 2ask 0.0007890625
-2dB 2fsk 0.0008203125
-2dB bpsk 1.5625e-05
0dB 2ask 7.8125e-06
0dB 2fsk 2.34375e-05
0dB bpsk 0.0
2dB 2ask 0.0
2dB 2fsk 0.0
2dB bpsk 0.0
4dB 2ask 0.0
4dB 2fsk 2.34375e-05
4dB bpsk 0.0
6dB 2ask 0.0
6dB 2fsk 0.0
6dB bpsk 0.0
8dB 2ask 0.0
8dB 2fsk 0.0
8dB bpsk 0.0
10dB 2ask 0.0
10dB 2fsk 0.0
10dB bpsk 0.0
    """.replace("dB", "")

    target_2ask = [0.1034,0.0553,0.0221,0.0058,0.0007,0.0000,0,0,0,0,0]
    target_bpsk = [0.0370,0.0121,0.0024,0.0002,0.0000,0,0,0,0,0,0]
    target_2fsk = [0.1849,0.1058,0.0449,0.0118,0.0014,0.0001,0,0,0,0,0]

    signal_type_ser = defaultdict(list)
    for line in info_list.split('\n'):
        info = line.strip().split(' ')
        if info == [""]:
            continue
        signal_type_ser[info[1]].append((info[0], info[2]))
    plt.semilogy(*zip(*signal_type_ser['bpsk']),label='model bpsk')
    plt.semilogy(*zip(*signal_type_ser['2ask']),label='model 2ask')
    plt.semilogy(*zip(*signal_type_ser['2fsk']),label='model 2fsk')
    plt.semilogy(range(-10, 12, 2), target_bpsk, label='target bpsk')
    plt.semilogy(range(-10, 12, 2), target_2ask, label='target 2ask')
    plt.semilogy(range(-10, 12, 2), target_2fsk, label='target 2fsk')
    plt.xlabel('SNR(dB)')
    plt.ylabel('SER')
    plt.title('Demodulation SER of BPSK, 2ASK and 2FSK')
    plt.legend(labels=['model bpsk', 'model 2ask', 'model 2fsk',
                       'target bpsk', 'target 2ask', 'target 2fsk'])
    plt.show()



if __name__ == '__main__':
    drawSERCurve()

