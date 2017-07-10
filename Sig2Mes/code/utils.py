# -*- encoding:utf-8 -*-
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
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
        return np.asarray(signals), np.asarray(messages), \
                np.asarray(signal_types,dtype=np.str), \
                np.asarray(dbs, dtype=np.str)


def loadMixSignal(filepath, M, sample_num=None):
    type2num = {'2ASK':0, '2FSK':1, 'BPSK':2}
    num2type = dict(zip(type2num.values(), type2num.keys()))
    snr2act_snr = dict()
    with open(filepath) as fr:
        signals = list()
        types = list()
        messages = list()
        snrs = list()
        for i, line in enumerate(fr):
            if i == sample_num:
                break
            signal, message, snr, act_snr, type = line.strip().split('\t')
            signal = map(float, signal.split(','))
            message = map(float, message.split(','))
            signal = np.asarray(signal)
            signal = signal.reshape(signal.shape[0], 1)
            message = np.asarray(map(lambda val: oneHot(val, M), message))
            snr = float(snr)
            act_snr = float(act_snr)
            type = oneHot(type2num[type], 3)
            signals.append(signal)
            messages.append(message)
            types.append(type)
            snrs.append(snr)
            snr2act_snr[snr] = act_snr
        return np.asarray(signals), np.asarray(messages), np.asarray(types), np.asarray(snrs), \
               snr2act_snr, num2type


def loadSymbol(filepath, M, sample_num):
    with open(filepath) as fr:
        symbols = list()
        messages = list()
        signal_types = list()
        dbs = list()
        for i, line in enumerate(fr):
            if i == sample_num:
                break
            info = line.strip().split('\t')
            if len(info) == 2:
                signal, message = info[0], info[1]
                st, db = 0, 0
            else:
                signal, message, st, db = info
            signal = map(float, signal.split(','))
            message = map(float, message.split(','))
            symbol = np.split(np.asarray(signal), 32)
            for sym, label in zip(symbol, message):
                sym = sym.reshape(sym.shape[0], 1)
                label = oneHot(label, M)
                symbols.append(sym)
                messages.append(label)
                signal_types.append(st)
                dbs.append(db)
        return np.asarray(symbols), np.asarray(messages), \
               np.asarray(signal_types,dtype=np.str), \
               np.asarray(dbs, dtype=np.str)


def loadMixSymbol(filepath, M, sample_num):
    type2num = {'2ASK':0, '2FSK':1, 'BPSK':2}
    num2type = dict(zip(type2num.values(), type2num.keys()))
    snr2act_snr = dict()
    with open(filepath) as fr:
        signals = list()
        types = list()
        messages = list()
        snrs = list()
        for i, line in enumerate(fr):
            if i == sample_num:
                break
            signal, message, snr, act_snr, type = line.strip().split('\t')
            signal = map(float, signal.split(','))
            signal = np.asarray(signal)
            signal = signal.reshape(signal.shape[0], 1)
            signal = np.split(signal, 32)
            message = map(float, message.split(','))
            message = np.asarray(map(lambda val: oneHot(val, M), message))
            message = np.split(message, 32)
            signals.extend(signal)
            messages.extend(message)
            snr = float(snr)
            act_snr = float(act_snr)
            type = oneHot(type2num[type], 3)
            types.extend([type for _ in xrange(32)])
            snrs.extend([snr for _ in xrange(32)])
            snr2act_snr[snr] = act_snr
        signals = np.asarray(signals)
        messages = np.asarray(messages)
        messages = messages.reshape(-1, messages.shape[-1])
        types = np.asarray(types)
        snrs = np.asarray(snrs)
        return signals, messages, types, snrs, snr2act_snr, num2type


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
    # info_list的模型输入为整个序列，输出为整个序列的解码
    info_list = """
10dB 2ASK 0.0
4dB 2FSK 0.0
-8dB 2ASK 0.0581883205351
0dB 2FSK 8.70914618698e-05
-6dB 2ASK 0.0236447334535
8dB 2ASK 0.0
-2dB 2FSK 0.00110979816045
4dB 2ASK 0.0
-4dB 2FSK 0.00741150442478
-10dB 2ASK 0.106705616167
0dB BPSK 0.0
-2dB BPSK 3.95970603142e-05
2dB 2FSK 0.0
0dB 2ASK 4.00435674013e-05
6dB BPSK 0.0
6dB 2FSK 0.0
-2dB 2ASK 0.000893289728682
2dB BPSK 0.0
-6dB BPSK 0.00292944207087
10dB BPSK 0.0
-8dB 2FSK 0.0765897755611
6dB 2ASK 0.0
-8dB BPSK 0.0139673913043
-10dB BPSK 0.0393949468085
-10dB 2FSK 0.144977706606
8dB 2FSK 0.0
8dB BPSK 0.0
2dB 2ASK 0.0
-4dB BPSK 0.000404631474104
10dB 2FSK 0.0
4dB BPSK 0.0
-6dB 2FSK 0.0297803971735
-4dB 2ASK 0.00612238697132
    """.replace("dB", "")

    # info_list2的输入为单个符号，输出为单个符号码字
    info_list2 = """
10dB 2ASK 0.0
4dB 2FSK 0.0
-8dB 2ASK 0.0551564401523
0dB 2FSK 3.89939559368e-05
-6dB 2ASK 0.0230138469869
8dB 2ASK 0.0
-2dB 2FSK 0.00177825885264
4dB 2ASK 0.0
-10dB 2FSK 0.189831304552
-4dB 2ASK 0.00581349673973
0dB BPSK 0.0
-2dB BPSK 0.0
2dB 2FSK 0.0
0dB 2ASK 3.90274362877e-05
6dB BPSK 0.0
6dB 2FSK 0.0
-2dB 2ASK 0.000782748229032
2dB BPSK 0.0
-6dB BPSK 0.00223687308688
10dB BPSK 0.0
-8dB 2FSK 0.10715813212
6dB 2ASK 0.0
-8dB BPSK 0.0115068065686
-10dB BPSK 0.0379122813882
-4dB 2FSK 0.0123244929797
8dB 2FSK 0.0
8dB BPSK 0.0
2dB 2ASK 0.0
-4dB BPSK 0.000157059839799
10dB 2FSK 0.0
4dB BPSK 0.0
-6dB 2FSK 0.0483908856002
-10dB 2ASK 0.105903118258
    """.replace("dB", "")

    target_2ask = [0.1034,0.0553,0.0221,0.0058,0.0007,0.0000,0,0,0,0,0]
    target_bpsk = [0.0370,0.0121,0.0024,0.0002,0.0000,0,0,0,0,0,0]
    target_2fsk = [0.1849,0.1058,0.0449,0.0118,0.0014,0.0001,0,0,0,0,0]

    signal_type_ser = defaultdict(list)
    for i, line in enumerate(info_list.split('\n')):
        info = line.strip().split(' ')
        if info == [""]:
            continue
        signal_type_ser[info[1]].append([int(info[0]), float(info[2])])

    symbol_type_ser = defaultdict(list)
    for i, line in enumerate(info_list2.split('\n')):
        info = line.strip().split(' ')
        if info == [""]:
            continue
        symbol_type_ser[info[1]].append([int(info[0]), float(info[2])])

    combined_type_ser = deepcopy(signal_type_ser)
    for i, line in enumerate(info_list2.split('\n')):
        info = line.strip().split(' ')
        if info == [""]:
            continue
        for _ in signal_type_ser:
            for __ in xrange(len(signal_type_ser[_])):
                if _ == info[1] and signal_type_ser[_][__][0] == int(info[0]):
                    combined_type_ser[info[1]][__][1] = \
                                min(signal_type_ser[info[1]][__][1], float(info[2]))

    signal_type_ser = dict(map(lambda key: (key, sorted(signal_type_ser[key], key=lambda kk:kk[0])), signal_type_ser))
    symbol_type_ser = dict(map(lambda key: (key, sorted(symbol_type_ser[key], key=lambda kk:kk[0])), symbol_type_ser))
    combined_type_ser = dict(map(lambda key: (key, sorted(combined_type_ser[key], key=lambda kk:kk[0])), combined_type_ser))

    total_labels = list()
    compare_models = ''
    if 1:
        if compare_models == '':
            compare_models += 'sequential model demodulation'
        else:
            compare_models += ' and sequential model demodulation'
        total_labels.append('Sequential Model Demodulation of BPSK')
        total_labels.append('Sequential Model Demodulation of 2ASK')
        total_labels.append('Sequential Model Demodulation of 2FSK')
        plt.semilogy(*zip(*signal_type_ser['BPSK']),label='Sequential Model Demodulation of BPSK')
        plt.semilogy(*zip(*signal_type_ser['2ASK']),label='Sequential Model Demodulation of 2ASK')
        plt.semilogy(*zip(*signal_type_ser['2FSK']),label='Sequential Model Demodulation of 2FSK')
    if 1:
        if compare_models == '':
            compare_models += 'symbolic model demodulation'
        else:
            compare_models += ' and symbolic model demodulation'
        total_labels.append('Symbolic Model Demodulation of BPSK')
        total_labels.append('Symbolic Model Demodulation of 2ASK')
        total_labels.append('Symbolic Model Demodulation of 2FSK')
        plt.semilogy(*zip(*symbol_type_ser['BPSK']),label='Symbolic Model Demodulation of BPSK')
        plt.semilogy(*zip(*symbol_type_ser['2ASK']),label='Symbolic Model Demodulation of 2ASK')
        plt.semilogy(*zip(*symbol_type_ser['2FSK']),label='Symbolic Model Demodulation of 2FSK')
    if 0:
        if compare_models == '':
            compare_models += 'combined model demodulation'
        else:
            compare_models += ' and combined model demodulation'
        total_labels.append('Combined Model Demodulation of BPSK')
        total_labels.append('Combined Model Demodulation of 2ASK')
        total_labels.append('Combined Model Demodulation of 2FSK')
        plt.semilogy(*zip(*combined_type_ser['BPSK']),label='Combined Model Demodulation of BPSK')
        plt.semilogy(*zip(*combined_type_ser['2ASK']),label='Combined Model Demodulation of 2ASK')
        plt.semilogy(*zip(*combined_type_ser['2FSK']),label='Combined Model Demodulation of 2FSK')
    if 0:
        if compare_models == '':
            compare_models += 'correlation demodulation'
        else:
            compare_models += ' and correlation demodulation'
        total_labels.append('Correlation Demodulation of BPSK')
        total_labels.append('Correlation Demodulation of 2ASK')
        total_labels.append('Correlation Demodulation of 2FSK')
        plt.semilogy(range(-10, 12, 2), target_bpsk, label='Correlation Demodulation of BPSK')
        plt.semilogy(range(-10, 12, 2), target_2ask, label='Correlation Demodulation of 2ASK')
        plt.semilogy(range(-10, 12, 2), target_2fsk, label='Correlation Demodulation of 2FSK')
    plt.xlabel('SNR(dB)')
    plt.ylabel('SER')
    plt.title('SER comparison of {} for modulated signals \n'
              'with BPSK, 2ASK and 2FSK respectively'.format(compare_models))
    plt.legend(labels=total_labels)
    plt.show()


def drawMixSignal(filepath):
    with open(filepath) as fr:
        for line in fr:
            signal = line.strip().split('\t')[0]
            signal = signal.split(',')
            signal = map(float, signal)
            message = line.strip().split('\t')[2]
            message = message.split(',')
            message = map(int, message)
            plt.subplot(2,1,1)
            plt.plot(signal)
            plt.subplot(2,1,2)
            plt.scatter(range(len(message)), message)
            break
        plt.show()


if __name__ == '__main__':
    drawSERCurve()
    # drawMixSignal('../data/10dB/mix_60000.txt')
