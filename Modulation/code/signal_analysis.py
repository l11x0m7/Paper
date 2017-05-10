# -*- encoding:utf-8 -*-
import os
import sys
import logging
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
reload(sys)


def drawModulation(dirpath, rownum=200):
    """信号文件绘图

    :param filepath: 需要显示绘图的信号文件路径
    :return: None
    """

    plt.figure(1)
    filepaths = os.listdir(dirpath)
    fileorder = 1
    useful_filepaths = [f for f in filepaths if f.startswith('parse_mod')]
    for filepath in useful_filepaths:
        count = np.random.randint(1, rownum + 1)
        with open(dirpath + '/' + filepath, 'rb') as fr:
            x = list()
            vals = list()
            name = filepath
            for i, line in enumerate(fr):
                if i < count:
                    continue
                if i > count:
                    break
                vals = line.strip().split('\t')
                vals = map(float, vals)
                x = range(len(vals))
            plt.subplot(2 * len(useful_filepaths), 1, fileorder * 2 - 1)
            plt.plot(x, vals, color = ((fileorder * 20 + 25) % 255 / 255.,
                                         (fileorder * 5 + 35) % 255 / 255.,
                                         (fileorder * 30 + 45) % 255 / 255.))
            plt.xlabel('symbol number')
            plt.ylabel('signal amplitude')
            plt.title(name)
        fileorder += 1
    plt.show()


def drawMixSignal(filepath, sample=5):
    """信号文件绘图

    :param filepath: 需要显示绘图的信号文件路径
    :return: None
    """

    plt.figure(1)
    with open(filepath, 'rb') as fr:
        rowNumber = sum(1 for _ in fr)
    with open(filepath, 'rb') as fr:
        sampleSignals = set(np.random.choice(range(rowNumber), sample, replace=False))
        rowOrder = 1
        for i, line in enumerate(fr):
            if i not in sampleSignals:
                continue
            vals = line.strip().split('\t')
            vals = map(float, vals)
            x = range(len(vals))
            plt.subplot(sample, 1, rowOrder)
            plt.plot(x, vals, color = ((rowOrder * 20 + 25) % 255 / 255.,
                                         (rowOrder * 5 + 35) % 255 / 255.,
                                         (rowOrder * 30 + 45) % 255 / 255.))
            rowOrder += 1
    plt.show()


def mixSignalAndTagging(dirpath='../data', savepath='../data/mixSignals.txt', modeSize=[]):
    """信号混叠和标注

    对已有的信号进行混叠.
    1-7分别对应：2ASK、QPSK、2FSK、2ASK+QPSK、2ASK+2FSK、QPSK+2FSK、2ASK+QPSK+2FSK

    :param dirpath: signal path
    :param modeSize: the sample size in each mode, from `1` to `n`
    :return: mixed signal
    """

    def tagger(tag):
        """

        给样本打标签,目前手动填写标签类型

        :param tag: like `1\t2`, `0\t2`, `0\t1\t2`
        :return: `int` from 1 to 7 representing label
        """

        if tag == '\t'.join(['0', ]):
            return 1
        elif tag == '\t'.join(['1', ]):
            return 2
        elif tag == '\t'.join(['2', ]):
            return 3
        elif tag == '\t'.join(['0', '1']):
            return 4
        elif tag == '\t'.join(['0', '2']):
            return 5
        elif tag == '\t'.join(['1', '2']):
            return 6
        elif tag == '\t'.join(['0', '1', '2']):
            return 7

    def C(n, m):
        def calcNext(count, point, l, r, res, pre):
            if(point > r):
                return
            if count == 1:
                for i in xrange(point, r + 1):
                    pre.append(i)
                    res.append(deepcopy(pre))
                    pre.pop()
            else:
                for i in xrange(point, r + 1):
                    pre.append(i)
                    calcNext(count - 1, i + 1, l, r, res, pre)
                    pre.pop()
        res = list()
        calcNext(m, 0, 0, n - 1, res, [])
        return res

    files = os.listdir(dirpath)
    signals = {}
    for filepath in files:
        if not filepath.startswith('parse_'):
            continue
        with open(dirpath + '/' + filepath, 'rb') as fr:
            modName = filepath.split('parse_mod_')[1].split('.txt')[0]
            signal = list()
            for line in fr:
                amps = line.strip().split('\t')
                amps = map(float, amps)
                signal.append(amps)
            # signal = zip(*signal)
            # signal = np.tile(signal, (20, 1))
            signals[modName] = signal

    modTypes = np.asarray(signals.keys())
    modeNum = len(modTypes)
    totalSignals = np.array([])
    totalTags = list()
    for mixNum in xrange(1, modeNum + 1):
        groupIndeces = C(modeNum, mixNum)
        groupNum = len(groupIndeces)
        sampleEachMod = modeSize[mixNum - 1] // groupNum
        groupSignals = np.array([])
        for groupInd in groupIndeces:
            mixSignals = np.array([])
            tag = '\t'.join(map(str, sorted(groupInd)))
            tag = str(tagger(tag))
            while len(mixSignals) < sampleEachMod:
                mixSignal = np.zeros([len(signals[modTypes[0]]), len(signals[modTypes[0]][0])])
                for ind in groupInd:
                    curSignal = np.asarray(signals[modTypes[ind]])
                    randomIndeces = np.random.choice(len(curSignal), len(curSignal), replace=False)
                    randSignal = curSignal[randomIndeces]
                    mixSignal += randSignal
                mixSignals = np.concatenate([mixSignals, mixSignal]) if mixSignals.shape[0] != 0 else mixSignal
            mixSignals = mixSignals[:sampleEachMod, :]
            totalTags.extend([tag] * sampleEachMod)
            groupSignals = np.concatenate([groupSignals, mixSignals]) if groupSignals.shape[0] != 0 else mixSignals
        totalSignals = np.concatenate([totalSignals, groupSignals]) if totalSignals.shape[0] != 0 else groupSignals

    assert len(totalTags) == sum(modeSize)
    assert len(totalSignals) == sum(modeSize)

    indeces = np.random.choice(len(totalSignals), len(totalSignals), replace=False)
    totalSignals = np.asarray(totalSignals)[indeces]
    totalTags = np.asarray(totalTags)[indeces]
    with open(savepath, 'wb') as fw:
        for i in xrange(len(totalTags)):
            signal = totalSignals[i]
            signal = map(str, signal)
            tag = totalTags[i]
            fw.write('\t'.join(['\t'.join(signal), tag]) + '\n')


def split(filepath):
    with open(filepath, 'rb') as fr:
        X = list()
        for line in fr:
            X.append(line.strip())
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    filename = filepath.split('/')[-1]
    dirbase = filepath.split('/')[:-1]
    with open('/'.join(dirbase + ['train_' + filename]), 'wb') as fw:
        for line in X_train:
            fw.write(line + '\n')
    with open('/'.join(dirbase + ['test_' + filename]), 'wb') as fw:
        for line in X_test:
            fw.write(line + '\n')


if __name__ == '__main__':
    # drawModulation('../data/5dB')
    drawMixSignal('../data/50dB/mixSignals.txt')
    # mixSignalAndTagging('../data/5dB', '../data/5dB/mixSignals.txt', [600, 1500, 2000])
    # split('../data/5dB/mixSignals.txt')
