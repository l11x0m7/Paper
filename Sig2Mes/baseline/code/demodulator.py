# -*- encoding=utf-8 -*-

import sys
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


# M = 2; Fs = 8Hz; Fc = 16Hz; Nsamp=32
def ask_demodulator(y, M, Fs, Fc, Nsamp):
    assert M == 2
    y = np.asarray(y)
    t_sin = 1. / Fs / Nsamp * np.arange(0, Nsamp)
    carrier = np.sin(2 * np.pi * Fc * t_sin)
    if y.ndim == 1:
        y = y.reshape((1, -1))
    m, n = y.shape
    res = np.zeros((m, n / Nsamp))
    for symbol in xrange(0, m):
        for i in xrange(0, n, Nsamp):
            z = np.sum(y[symbol, i:i+Nsamp] * carrier) / float(Fc)
            if z > 0.5:
                res[symbol, (i+Nsamp-1) / Nsamp] = 1
    return res


# M = 2; freq_sep = 4; Nsamp = 32; Fs = 8; Fc = 16
def fsk_demodulator(y, M, freq_sep, Nsamp, Fs, Fc):
    assert M == 2
    y = np.asarray(y)
    f1 = Fc - freq_sep
    f2 = Fc + freq_sep
    t_sin = 1. / Fs / Nsamp * np.arange(0, Nsamp)
    carrier1 = np.cos(2*np.pi*f1*t_sin)
    carrier2 = np.cos(2*np.pi*f2*t_sin)
    if y.ndim == 1:
        y = y.reshape((1, -1))
    m, n = y.shape
    res = np.zeros((m, n / Nsamp))
    for symbol in xrange(0, m):
        for i in xrange(0, n, Nsamp):
            z1 = abs(np.sum(y[symbol,i:i+Nsamp] * carrier1) / Fc)
            z2 = abs(np.sum(y[symbol,i:i+Nsamp] * carrier2) / Fc)
            if z1 >= z2:
                res[symbol, (i+Nsamp-1) / Nsamp] = 1
            elif z1 < z2:
                res[symbol, (i+Nsamp-1) / Nsamp] = 0
    return res


# M = 2; Fs = 8Hz; Fc = 16Hz; Nsamp=32
def psk_demodulator(y, M, Fs, Fc, Nsamp):
    assert M == 2
    y = np.asarray(y)
    t_sin = 1. / Fs / Nsamp * np.arange(0, Nsamp)
    carrier = np.sin(2 * np.pi * Fc * t_sin)
    if y.ndim == 1:
        y = y.reshape((1, -1))
    m, n = y.shape
    res = np.zeros((m, n / Nsamp))
    for symbol in xrange(0, m):
        for i in xrange(0, n, Nsamp):
            z = np.sum(y[symbol, i:i+Nsamp] * carrier) / Fc
            if z < 0:
                res[symbol, (i+Nsamp-1) / Nsamp] = 1
    return res


def ser(y1, y2):
    return float(np.sum(np.not_equal(y1, y2))) / np.prod(np.shape(y1))


def test_2ask():
    signals = list()
    messages = list()
    with open('../../data/10dB/2ask_20000.txt', 'rb') as fr:
        for line in fr:
            items = line.strip().split('\t')
            signal = map(float, items[0].split(','))
            message = map(int, items[1].split(','))
            signals.append(signal)
            messages.append(message)
    signals = np.asarray(signals)
    messages = np.asarray(messages)
    demod_signals = ask_demodulator(signals, 2, 8, 16, 32)
    ber_value = ser(messages, demod_signals)
    print(ber_value)


def test_2fsk():
    signals = list()
    messages = list()
    with open('../../data/10dB/2fsk_20000.txt', 'rb') as fr:
        for line in fr:
            items = line.strip().split('\t')
            signal = map(float, items[0].split(','))
            message = map(int, items[1].split(','))
            signals.append(signal)
            messages.append(message)
    signals = np.asarray(signals)
    messages = np.asarray(messages)
    demod_signals = fsk_demodulator(signals, 2, 4, 32, 8, 16)
    ber_value = ser(messages, demod_signals)
    print(ber_value)


def test_bpsk():
    signals = list()
    messages = list()
    with open('../../data/10dB/bpsk_20000.txt', 'rb') as fr:
        for line in fr:
            items = line.strip().split('\t')
            signal = map(float, items[0].split(','))
            message = map(int, items[1].split(','))
            signals.append(signal)
            messages.append(message)
    signals = np.asarray(signals)
    messages = np.asarray(messages)
    demod_signals = psk_demodulator(signals, 2, 8, 16, 32)
    ber_value = ser(messages, demod_signals)
    print(ber_value)


def test_mix_signals():
    # 相关解调结果
    # res = defaultdict(dict)
    # x_label = set()
    # with open('../../data/mix_data/mix_-10_20_210000.txt', 'rb') as fr:
    #     for line in fr:
    #         items = line.strip().split('\t')
    #         signal = map(float, items[0].split(','))
    #         message = map(int, items[1].split(','))
    #         snr, act_snr, st = float(items[2]), float(items[3]), items[4]
    #         x_label.add((snr, act_snr))
    #         res[st].setdefault(snr, [0, 0])
    #         if st == '2ASK':
    #             demod_signal = ask_demodulator(signal, 2, 8, 16, 32)
    #             res[st][snr][0] += np.sum(np.not_equal(demod_signal, message))
    #             res[st][snr][1] += np.prod(demod_signal.shape)
    #         elif st == '2FSK':
    #             demod_signal = fsk_demodulator(signal, 2, 4, 32, 8, 16)
    #             res[st][snr][0] += np.sum(np.not_equal(demod_signal, message))
    #             res[st][snr][1] += np.prod(demod_signal.shape)
    #         elif st == 'BPSK':
    #             demod_signal = psk_demodulator(signal, 2, 8, 16, 32)
    #             res[st][snr][0] += np.sum(np.not_equal(demod_signal, message))
    #             res[st][snr][1] += np.prod(demod_signal.shape)
    #         else:
    #             raise ValueError('Wrong signal type!')

        x_label = set([(20.0, -3.1), (-10.0, -15.1), (-5.0, -10.6), (0.0, -7.0), (10.0, -3.6), (5.0, -4.7), (15.0, -3.2)])
        res= {'BPSK': {0.0: [59935, 320000],
                       5.0: [47299, 320000],
                       10.0: [41405, 320000],
                       15.0: [39995, 320000],
                       20.0: [40074, 320000],
                       -10.0: [84622, 320000],
                       -5.0: [73237, 320000]},
              '2ASK': {0.0: [93918, 320000],
                       5.0: [88687, 320000],
                       10.0: [84886, 320000],
                       15.0: [82097, 320000],
                       20.0: [80837, 320000],
                       -10.0: [107327, 320000],
                       -5.0: [100795, 320000]},
              '2FSK': {0.0: [107452, 320000],
                       5.0: [111889, 320000],
                       10.0: [118582, 320000],
                       15.0: [119862, 320000],
                       20.0: [119546, 320000],
                       -10.0: [131699, 320000],
                       -5.0: [114997, 320000]}}

        x_label = sorted(list(x_label), key=lambda k:k[0])
        x_label = {float(r[0]):str(list(r)) for r in x_label}
        def format_fn(tick_val, tick_pos):
            if int(tick_val) in x_label:
                return x_label[int(tick_val)]
            else:
                return ''
        plt.figure()
        ax = plt.subplot(111)
        ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        st_list = list()
        compare_models = ''

        # for i, st in enumerate(res):
        #     st_label = 'Correlation Demodulation of ' + st
        #     if i == 0:
        #         if compare_models == '':
        #             compare_models += 'correlation demodulation'
        #         else:
        #             compare_models += ' and correlation demodulation'
        #     st_list.append(st_label)
        #     snr2ser = dict()
        #     for snr in res[st]:
        #         ser = res[st][snr][0] / float(res[st][snr][1])
        #         snr2ser[snr] = ser
        #         print('{}\t{}\t{}'.format(st, snr, ser))
        #     sers = zip(*sorted(snr2ser.iteritems(), key=lambda k:k[0]))[1]
        #     plt.semilogy(range(-10, 25, 5), sers, label=st_label)

        # 模型结果（sequential model）
        signal_model_snr = """
BPSK	0.0	0.129629060039
BPSK	5.0	0.0539730367793
BPSK	10.0	0.00709667487685
BPSK	15.0	0.000140625
BPSK	20.0	0.0
BPSK	-10.0	0.251125562781
BPSK	-5.0	0.19194828469
2ASK	0.0	0.263715277778
2ASK	5.0	0.13231180061
2ASK	10.0	0.034541868932
2ASK	15.0	0.00292721518987
2ASK	20.0	0.0
2ASK	-10.0	0.345221480583
2ASK	-5.0	0.335833333333
2FSK	0.0	0.148786618669
2FSK	5.0	0.0449860900354
2FSK	10.0	0.00283511722732
2FSK	15.0	0.0
2FSK	20.0	0.0
2FSK	-10.0	0.388075906344
2FSK	-5.0	0.27523091133
        """
        # symbolic model
        symbol_model_snr = """
BPSK	0.0	0.150248383166
BPSK	5.0	0.0340044882184
BPSK	10.0	0.00167436037869
BPSK	15.0	0.0
BPSK	20.0	0.0
BPSK	-10.0	0.242444593687
BPSK	-5.0	0.191598746082
2ASK	0.0	0.283181254221
2ASK	5.0	0.0930435596021
2ASK	10.0	0.00774805515989
2ASK	15.0	0.000249112536588
2ASK	20.0	0.0
2ASK	-10.0	0.333583948123
2ASK	-5.0	0.320145070887
2FSK	0.0	0.176915133308
2FSK	5.0	0.0492611606098
2FSK	10.0	0.00276938963904
2FSK	15.0	1.56067108857e-05
2FSK	20.0	0.0
2FSK	-10.0	0.418979966975
2FSK	-5.0	0.320570207589
        """

        sequential_st_snr = defaultdict(list)
        for line in signal_model_snr.strip().split('\n'):
            items = line.split('\t')
            sequential_st_snr[items[0]].append((float(items[1]), float(items[2])))

        symbol_st_snr = defaultdict(list)
        for line in symbol_model_snr.strip().split('\n'):
            items = line.split('\t')
            symbol_st_snr[items[0]].append((float(items[1]), float(items[2])))

        for i, st in enumerate(sequential_st_snr):
            st_label = 'Sequential Model Demodulation of ' + st
            st_list.append(st_label)
            if i == 0:
                if compare_models == '':
                    compare_models += 'sequential model demodulation'
                else:
                    compare_models += ' and sequential model demodulation'
            sers = zip(*sorted(sequential_st_snr[st], key=lambda k: k[0]))[1]
            plt.semilogy(range(-10, 25, 5), sers, label=st_label)

        for i, st in enumerate(symbol_st_snr):
            if i == 0:
                st_label = 'Symbolic Model Demodulation of ' + st
                if compare_models == '':
                    compare_models += 'symbolic model demodulation'
                else:
                    compare_models += ' and symbolic model demodulation'
            st_list.append(st_label)
            sers = zip(*sorted(symbol_st_snr[st], key=lambda k: k[0]))[1]
            plt.semilogy(range(-10, 25, 5), sers, label=st_label)

        # for i, st in enumerate(symbol_st_snr):
        #     st_label = 'Combined Model Demodulation of ' + st
        #     if i == 0:
        #         if compare_models == '':
        #             compare_models += 'combined model demodulation'
        #         else:
        #             compare_models += ' and combined model demodulation'
        #     st_list.append(st_label)
        #     sequential_sers = zip(*sorted(sequential_st_snr[st], key=lambda k: k[0]))[1]
        #     symbolic_sers = zip(*sorted(symbol_st_snr[st], key=lambda k: k[0]))[1]
        #     combined_sers = [min(symbolic_ser, sequential_ser)
        #                      for (symbolic_ser, sequential_ser) in
        #                      zip(sequential_sers, symbolic_sers)]
        #     plt.semilogy(range(-10, 25, 5), combined_sers, label=st_label)

        plt.xticks(range(-10, 25, 5))
        plt.xlabel('[SNR(dB), Actual SNR(dB)]')
        plt.ylabel('SER')
        plt.title('SER comparison of {} for composite signals \n'
                  'modulated with BPSK, 2ASK and 2FSK respectively'.format(compare_models))
        plt.legend(labels=st_list)
        plt.show()




if __name__ == '__main__':
    # test_2ask()
    # test_2fsk()
    # test_bpsk()
    test_mix_signals()
