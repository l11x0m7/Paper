import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import json

label_transformer = {'QPSK': 0, 'AM-DSB': 1, 'AM-SSB': 2,
                     'PAM4': 3, 'WBFM': 4, 'GFSK': 5, 'CPFSK': 6,
                     'QAM16': 7, 'QAM64': 8, 'BPSK': 9, '8PSK': 10}

inverse_label_transformer = dict(zip(label_transformer.values(), label_transformer.keys()))

def rmlParser(filepath):
    with open(filepath, 'rb') as fr:
        array = pickle.load(fr)
        modulation, snr = zip(*array.keys())
        print set(modulation)
        print set(snr)


def getRealSignal(filepath, savepath):
    fw = open(savepath, 'wb')
    with open(filepath, 'rb') as fr:
        array = pickle.load(fr)
        for each_type in array:
            array[each_type] = array[each_type][:,0,:]
        pickle.dump(array, fw)


def mixSignal(filepath, savepath):
    raise NotImplementedError


def drawSignal(filepath):
    with open(filepath, 'rb') as fr:
        pre_snr = None
        pre_label = None
        plt.figure()
        nrow = 3
        ncol = 4
        count = 0
        for line in fr:
            items = line.strip().split('\t')
            signal = json.loads(items[0])
            label = inverse_label_transformer[int(items[1])]
            snr = int(items[2])
            if label == 'AM-DSB' and snr == 18:
                fig = plt.subplot(nrow, ncol, count + 1)
                count += 1
                pre_label = label
                real, = plt.plot(signal[0], c='b')
                image, = plt.plot(signal[1], c='r')
                plt.xlabel('time')
                plt.ylabel('ampitude')
                plt.legend([real, image], ['real', 'image'])
                plt.title('{} {}'.format(label, snr))
                if count == nrow * ncol:
                    break
        plt.show()


def splitSignal(filepath):
    total_train = list()
    total_test = list()
    with open(filepath, 'rb') as fr:
        array = pickle.load(fr)
        for mode, snr in array.keys():
            label = label_transformer[mode]
            X_train, X_test = train_test_split(array[(mode, snr)], test_size=0.2)
            for i, train in enumerate(X_train):
                sample = '\t'.join([json.dumps(train.tolist()), str(label), str(snr)]) + '\n'
                total_train.append(sample)
            for i, test in enumerate(X_test):
                sample = '\t'.join([json.dumps(test.tolist()), str(label), str(snr)]) + '\n'
                total_test.append(sample)
            print '{} {} done!'.format(mode, snr)
    np.random.shuffle(total_train)
    np.random.shuffle(total_test)
    with open(filepath + '.train', 'wb') as fw:
        for line in total_train:
            fw.write(line)
    with open(filepath + '.test', 'wb') as fw:
        for line in total_test:
            fw.write(line)


if __name__ == '__main__':
    # rmlParser('../rml_data/RML2016.10a_dict.dat')
    # getRealSignal('../rml_data/RML2016.10a_dict.dat', '../rml_data/RML2016.10a_dict_real.dat')
    # mixSignal('../rml_data/RML2016.10a_dict_real.dat', '../rml_data/dict_mixSignal.dat')
    splitSignal('../rml_data/RML2016.10a_dict.dat')
    # drawSignal('../rml_data/RML2016.10a_dict.dat.train')