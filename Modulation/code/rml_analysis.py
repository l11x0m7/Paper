import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

label_transformer = {'QPSK': 0, 'AM-DSB': 1, 'AM-SSB': 2,
                     'PAM4': 3, 'WBFM': 4, 'GFSK': 5, 'CPFSK': 6,
                     'QAM16': 7, 'QAM64': 8, 'BPSK': 9, '8PSK': 10}

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


def splitSignal(filepath):
    fwtrain = open(filepath + '.train', 'wb')
    fwtest = open(filepath + '.test', 'wb')
    with open(filepath, 'rb') as fr:
        array = pickle.load(fr)
        for mode, snr in array.keys():
            label = label_transformer[mode]
            X_train, X_test = train_test_split(array[(mode, snr)], test_size=0.2)
            for i, train in enumerate(X_train):
                fwtrain.write('\t'.join(map(str, train.tolist() + [label])) + '\n')
            for i, test in enumerate(X_test):
                fwtest.write('\t'.join(map(str, test.tolist() + [label])) + '\n')
            print '{} {} done!'.format(mode, snr)
        fwtrain.close()
        fwtest.close()



if __name__ == '__main__':
    # rmlParser('../rml_data/RML2016.10a_dict.dat')
    # getRealSignal('../rml_data/RML2016.10a_dict.dat', '../rml_data/RML2016.10a_dict_real.dat')
    # mixSignal('../rml_data/RML2016.10a_dict_real.dat', '../rml_data/dict_mixSignal.dat')
    splitSignal('../rml_data/RML2016.10a_dict_real.dat')