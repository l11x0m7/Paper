# -*- encoding:utf-8 -*-
from model import *

"""
    实验目的：从混合信号中对相同载波不同调制方式且已知调制方式的调制信号解码，
    得到一个有用信号+一个或多个干扰信号。实际应用场景中需要对强干扰信号进行分析
    实验方法： 使用深度卷积网络从混合信号中解码出需要的信号序列
    实验数据： 调制类型、信噪比、混合信号、解码序列，信噪比-10到10，3种调制类型（2ASK、2FSK、BPSK），各1000个
    立足点:   1. 模型角度：抗干扰性能好；学习混合信号的统计特征，可以分别解调出混合信号中的大功率信号；
             2. 项目角度：需要从多个已知调制类型的同频混合信号中提取需要的信号（包括有用信号和干扰信号）；
             3. 传统方法：需要经过同步、分离、抗干扰处理、解调；难以从同频干扰中提取信号
"""


def mixSignalTest(filepath, M, sample_num):
    class LossHistory(keras.callbacks.Callback):
        def __init__(self, x, signal_type, y):
            self.x = x
            self.signal_type = signal_type
            self.y = y
            self.logs = {}
            super(LossHistory, self).__init__()

        def on_train_begin(self, logs={}):
            self.train_losses = []
            self.val_losses = []
            self.val_ser = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            res = model.predict([self.x, self.signal_type])
            ser = float(np.sum(np.argmax(res, axis=2) != np.argmax(self.y, axis=2))) \
                    / self.y.shape[0] / self.y.shape[1]
            self.val_ser.append(ser)
            logs['ser'] = ser
            self.logs = logs
            print '\nvalidation average error decode rate: ', ser

    input_length = 1024
    input_dim = 1
    input_depth = 1

    batch_size = 128

    X, y, signal_types, snrs, snr2act_snrs, num2type = loadMixSignal(filepath, M, sample_num)
    X = X[:,:,:,None]
    print X.shape, signal_types.shape, y.shape

    train_X, test_X, train_y, test_y, train_signal_type, test_signal_type, train_snr, test_snr \
        = train_test_split(X, y, signal_types, snrs, train_size=0.8)

    train_X, val_X, train_y, val_y, train_signal_type, val_signal_type, train_snr, val_snr \
        = train_test_split(train_X, train_y, train_signal_type, train_snr, train_size=0.9)

    model = MixSignalDecoder(filter_num=64, kernel_size=(32, 1), strides=(16, 1),
                             input_shape=(input_length, input_dim, input_depth),
                             dropout=0.5, label_size=M, signal_type_shape=(3,))
    print model.summary()
    plot_model(model, to_file='composite_signal_model.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    logCallBack = LossHistory(val_X, val_signal_type, val_y)
    model.fit([train_X, train_signal_type], train_y, epochs=50,
              batch_size=batch_size, verbose=1,
              callbacks=[logCallBack, EarlyStopping(patience=3, monitor='ser', verbose=0)])
    st2X = defaultdict(dict)
    st2y = defaultdict(dict)
    st2st = defaultdict(dict)
    for x, y, st, snr in zip(test_X, test_y, test_signal_type, test_snr):
        st_ind = num2type[np.argmax(st)]
        st2X[st_ind].setdefault(snr, list())
        st2y[st_ind].setdefault(snr, list())
        st2st[st_ind].setdefault(snr, list())
        st2X[st_ind][snr].append(x)
        st2y[st_ind][snr].append(y)
        st2st[st_ind][snr].append(st)

    print 'signal type\tsnr\taverage error decode rate'
    for st in st2X:
        for snr in st2X[st]:
            x = np.asarray(st2X[st][snr])
            y = np.asarray(st2y[st][snr])
            signal_type = np.asarray(st2st[st][snr])
            res = model.predict([x, signal_type])
            print('\t'.join(map(str, [st, snr,
                float(np.sum(np.argmax(res, axis=2) != np.argmax(y, axis=2))) \
                 / y.shape[0] / y.shape[1]])))


def mixSymbolTest(filepath, M, sample_num):
    class LossHistory(keras.callbacks.Callback):
        def __init__(self, x, signal_type, y):
            self.x = x
            self.signal_type = signal_type
            self.y = y
            self.logs = {}
            super(LossHistory, self).__init__()

        def on_train_begin(self, logs={}):
            self.train_losses = []
            self.val_losses = []
            self.val_ser = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            res = model.predict([self.x, self.signal_type])
            ser = float(np.sum(np.argmax(res, axis=1) != np.argmax(self.y, axis=1))) \
                  / self.y.shape[0]
            self.val_ser.append(ser)
            logs['ser'] = ser
            self.logs = logs
            print '\nvalidation average error decode rate: ', ser

    input_length = 32
    input_dim = 1
    input_depth = 1

    batch_size = 128

    X, y, signal_types, snrs, snr2act_snrs, num2type = \
                        loadMixSymbol(filepath, M, sample_num)
    X = X[:, :, :, None]
    print X.shape, signal_types.shape, y.shape

    train_X, test_X, train_y, test_y, train_signal_type, test_signal_type, train_snr, test_snr \
        = train_test_split(X, y, signal_types, snrs, train_size=0.8)

    train_X, val_X, train_y, val_y, train_signal_type, val_signal_type, train_snr, val_snr \
        = train_test_split(train_X, train_y, train_signal_type, train_snr, train_size=0.9)

    model = MixSymbolDemodulator(filter_num=64,
                             input_shape=(input_length, input_dim, input_depth),
                             dropout=0.5, label_size=M, signal_type_shape=(3,))
    print model.summary()
    plot_model(model, to_file='composite_symbol_model.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    logCallBack = LossHistory(val_X, val_signal_type, val_y)
    model.fit([train_X, train_signal_type], train_y, epochs=1,
              batch_size=batch_size, verbose=1,
              callbacks=[logCallBack, EarlyStopping(patience=3, monitor='ser', verbose=0)])

    st2X = defaultdict(dict)
    st2y = defaultdict(dict)
    st2st = defaultdict(dict)
    for x, y, st, snr in zip(test_X, test_y, test_signal_type, test_snr):
        st_ind = num2type[np.argmax(st)]
        st2X[st_ind].setdefault(snr, list())
        st2y[st_ind].setdefault(snr, list())
        st2st[st_ind].setdefault(snr, list())
        st2X[st_ind][snr].append(x)
        st2y[st_ind][snr].append(y)
        st2st[st_ind][snr].append(st)

    print 'signal type\tsnr\taverage error decode rate'
    for st in st2X:
        for snr in st2X[st]:
            x = np.asarray(st2X[st][snr])
            y = np.asarray(st2y[st][snr])
            signal_type = np.asarray(st2st[st][snr])
            res = model.predict([x, signal_type])
            print('\t'.join(map(str, [
                st, snr,
                float(np.sum(np.argmax(res, axis=1) != np.argmax(y, axis=1))) / y.shape[0]])))


if __name__ == '__main__':
    # mixSignalTest('../data/mix_data/mix_-10_20_210000.txt', 2, None)
    mixSymbolTest('../data/mix_data/mix_-10_20_4200.txt', 2, None)
    # args = sys.argv
    # filepath = args[1]
    # mixSignalTest(filepath, 4, 3, None)