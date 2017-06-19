from model import *


def mixSignalTest(filepath, M, mix_type, sample_num):
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
            self.test_losses = []
            self.test_ser = []

        def on_epoch_end(self, epoch, logs={}):
            self.train_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            res = model.predict([self.x, self.signal_type])
            ser = float(np.sum(np.argmax(res, axis=2) != np.argmax(self.y, axis=2))) \
                    / self.y.shape[0] / self.y.shape[1]
            test_loss = log_loss(np.argmax(self.y, axis=2).flatten(), np.concatenate(res, axis=0))
            self.test_ser.append(ser)
            self.test_losses.append(test_loss)
            self.logs = logs
            logs['ser'] = ser
            print '\ntest average error decode rate: ', ser
            print 'test cross entropy loss: {}'.format(test_loss)

    input_length = 960
    input_dim = 1
    input_depth = 1

    batch_size = 128

    X, signal_type, y = loadMixSignal(filepath, M, mix_type, sample_num)
    X = X[:,:,:,None]
    print X.shape, y.shape

    train_X, test_X, train_y, test_y, train_signal_type, test_signal_type \
        = train_test_split(X, y, signal_type, train_size=0.8)

    model = MixSignalDecoder(filter_num=64, kernel_size=(30, 1), strides=(15, 1),
                             input_shape=(input_length, input_dim, input_depth),
                             dropout=0.5, label_size=M, signal_type_shape=(3,))
    print model.summary()
    plot_model(model, to_file='mix_model.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    logCallBack = LossHistory(test_X, test_signal_type, test_y)
    model.fit([train_X, train_signal_type], train_y, epochs=50,
              batch_size=batch_size, validation_split=0.1, verbose=1,
              callbacks=[logCallBack, EarlyStopping(patience=3, monitor='ser', verbose=0)])
    print 'train loss: ', json.dumps(logCallBack.train_losses)
    print 'val loss: ', json.dumps(logCallBack.val_losses)
    print 'test loss: ', json.dumps(logCallBack.test_losses)
    print 'test SER: ', json.dumps(logCallBack.test_ser)
    # res = model.predict([X, signal_type])
    # print 'average error decode rate: ', float(np.sum(np.argmax(res, axis=2) != np.argmax(y, axis=2))) \
    #                                      / y.shape[0] / y.shape[1]


if __name__ == '__main__':
    mixSignalTest('../data/0dB/mix_60000.txt', 4, 3, None)
    # args = sys.argv
    # filepath = args[1]
    # mixSignalTest(filepath, 4, 3, None)