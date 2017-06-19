# -*- encoding=utf-8 -*-
from model import *
from keras.optimizers import Adadelta


def signalSimpleSeq2Seq(filepath, M, sample_num, db, signal_type):
    class LossHistory(keras.callbacks.Callback):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            super(LossHistory, self).__init__()

        def on_train_begin(self, logs={}):
            self.train_losses = []
            self.val_losses = []
            self.val_ser = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            res = model.predict(self.x)
            ser = float(np.sum(np.argmax(res, axis=2) != np.argmax(self.y, axis=2))) \
                    / self.y.shape[0] / self.y.shape[1]
            self.val_ser.append(ser)
            self.logs = logs
            logs['ser'] = ser
            print '\nvalidation average error decode rate: ', ser
    input_length = 1024
    input_dim = 1
    input_depth = 1

    output_length = 32
    output_dim = 16
    batch_size = 128

    X, y = loadSignal(filepath, M, sample_num)
    X = X[:,:,:,None]
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, train_size=0.9)

    # model = SimpleCNNSeq2Seq(filter_num=32, kernel_size=(30, 1), strides=(15, 1),
    #         output_dim=output_dim, output_length=output_length,
    #         input_shape=(input_length, input_dim, input_depth), depth=2,
    #         dropout=0.5, model_type='attention')
    # model = DeepCNNSeq2Seq(filter_num=64, kernel_size=(30, 1), strides=(15, 1), label_size=M,
    #                          output_dim=output_dim, output_length=output_length,
    #                          input_shape=(input_length, input_dim, input_depth), depth=1,
    #                          dropout=0.5, model_type='attention')
    model = DeepCNN(filter_num=64, kernel_size=(32, 1), strides=(16, 1),
                    input_shape=(input_length, input_dim, input_depth),
                    dropout=0.5, label_size=M)
    # model = MixSignalDecoder(filter_num=64, kernel_size=(30, 1), strides=(15, 1),
    #                 input_shape=(input_length, input_dim, input_depth),
    #                 dropout=0.5, label_size=M, signal_type_shape=(3,))
    # model = DeeperCNN(filter_num=32, kernel_size=(6, 1), strides=(3, 1),
    #                 input_shape=(input_length, input_dim, input_depth),
    #                 dropout=0.5, label_size=M)
    # print model.summary()
    # plot_model(model, to_file='single_model.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    logCallBack = LossHistory(valid_X, valid_y)
    model_checkpoint = keras.callbacks.ModelCheckpoint('../model/model.ckpt',
                            monitor='ser', verbose=0,
                            save_best_only=True, save_weights_only=False,
                            mode='auto', period=1)
    his = model.fit(train_X, train_y, epochs=50, batch_size=batch_size, verbose=0,
              callbacks=[logCallBack, EarlyStopping(patience=3, monitor='ser'),
                         model_checkpoint])
    model.load_weights('../model/model.ckpt')
    res = model.predict(test_X)
    print db, signal_type, float(np.sum(np.argmax(res, axis=2) !=
                                     np.argmax(test_y, axis=2)))\
                                     / test_y.shape[0] / test_y.shape[1]
    # cm = confusion_matrix(np.argmax(res, axis=2).flatten(),
    #                       np.argmax(test_y, axis=2).flatten())
    # save_confusion_matrix(cm,
    #             title='confusion matrix of {}, SNR={}'.format(signal_type, db),
    #             labels=range(M),
    #             savepath='../figure/{}_{}.png'.format(signal_type, db))


def signalGeneralSeq2Seq(filepath, M, sample_num):
    class LossHistory(keras.callbacks.Callback):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            super(LossHistory, self).__init__()

        def on_train_begin(self, logs={}):
            self.train_losses = []
            self.val_losses = []
            self.val_ser = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            res = model.predict(self.x)
            ser = float(np.sum(np.argmax(res, axis=2) != np.argmax(self.y, axis=2))) \
                  / self.y.shape[0] / self.y.shape[1]
            self.val_ser.append(ser)
            self.logs = logs
            logs['ser'] = ser
            print '\nvalidation average error decode rate: ', ser

    input_length = 1024
    input_dim = 1
    input_depth = 1

    output_length = 32
    output_dim = 16
    batch_size = 128

    X, y, signal_types, dbs = loadSignal(filepath, M, sample_num)
    X = X[:,:,:,None]
    train_X, test_X, train_y, test_y, train_st, test_st, train_db, test_db \
        = train_test_split(X, y, signal_types, dbs, train_size=0.8)
    train_X, valid_X, train_y, valid_y, train_st, valid_st, train_db, valid_db \
        = train_test_split(train_X, train_y, train_st, train_db, train_size=0.9)
    st_db_test_X = defaultdict(list)
    st_db_test_y = defaultdict(list)
    for x, y, st, db in zip(test_X, test_y, test_st, test_db):
        st_db_test_X[(st, db)].append(x)
        st_db_test_y[(st, db)].append(y)
    map(np.asarray, st_db_test_X)
    map(np.asarray, st_db_test_y)

    model = DeepCNN(filter_num=64, kernel_size=(32, 1), strides=(16, 1),
                    input_shape=(input_length, input_dim, input_depth),
                    dropout=0.5, label_size=M)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    logCallBack = LossHistory(valid_X, valid_y)
    model_checkpoint = keras.callbacks.ModelCheckpoint('../model/model.ckpt',
                                                       monitor='ser', verbose=0,
                                                       save_best_only=True, save_weights_only=False,
                                                       mode='auto', period=1)
    his = model.fit(train_X, train_y, epochs=200, batch_size=batch_size, verbose=0,
                    callbacks=[logCallBack, EarlyStopping(patience=3, monitor='ser'),
                               model_checkpoint])
    model.load_weights('../model/model.ckpt')
    for (st, db) in st_db_test_X:
        test_X = st_db_test_X[(st, db)]
        test_y = st_db_test_y[(st, db)]
        res = model.predict(test_X)
        print db, st, float(np.sum(np.argmax(res, axis=2) !=
                                        np.argmax(test_y, axis=2))) \
                           / test_y.shape[0] / test_y.shape[1]
    # cm = confusion_matrix(np.argmax(res, axis=2).flatten(),
    #                       np.argmax(test_y, axis=2).flatten())
    # save_confusion_matrix(cm,
    #             title='confusion matrix of {}, SNR={}'.format(signal_type, db),
    #             labels=range(M),
    #             savepath='../figure/{}_{}.png'.format(signal_type, db))


if __name__ == '__main__':
    # for db in ['-10dB', '-8dB', '-6dB', '-4dB', '-2dB', '0dB', '2dB',
    #            '4dB', '6dB', '8dB', '10dB']:
    #     for t in ['2ask_20000.txt', '2fsk_20000.txt', 'bpsk_20000.txt']:
    #         signal_type = t.split('_')[0]
    #         if t == 'qpsk_20000.txt':
    #             signalSimpleSeq2Seq('../data/{}/{}'.format(db, t), 4, None, db, signal_type)
    #         else:
    #             signalSimpleSeq2Seq('../data/{}/{}'.format(db, t), 2, None, db, signal_type)

    # 对每个信噪比下每种信号类型对应一个模型
    # signalSimpleSeq2Seq('../data/0dB/bpsk_20000.txt', 2, None, -10, 'BPSK')


    # 对所有信噪比下所有信号类型对应一个模型
    signalGeneralSeq2Seq('../data/merge_n10_10_660000_demo.txt', 2, None)