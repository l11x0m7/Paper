# -*- encoding=utf8 -*-
import os
import sys
import tensorflow as tf
import keras
from keras import backend as K
from keras.utils.test_utils import keras_test
from keras.layers import Recurrent
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Reshape, RepeatVector
from keras.layers import Lambda, TimeDistributed, Bidirectional, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Concatenate
from keras.activations import relu, softmax
from keras.models import Model, Sequential
from recurrentshop import RecurrentSequential, LSTMCell
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import json

from cells import LSTMDecoderCell, AttentionDecoderCell
from utils import *

def SimpleSeq2Seq(output_dim, output_length, hidden_dim=None, input_shape=None,
                  batch_size=None, batch_input_shape=None, input_dim=None,
                  input_length=None, depth=1, dropout=0.0, unroll=False,
                  stateful=False):

    '''
    Simple model for sequence to sequence learning.
    The encoder encodes the input sequence to vector (called context vector)
    The decoder decodes the context vector in to a sequence of vectors.
    There is no one on one relation between the input and output sequence
    elements. The input sequence and output sequence may differ in length.
    Arguments:
    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3,
            there will be 3 LSTMs on the enoding side and 3 LSTMs on the
            decoding side. You can also specify depth as a tuple. For example,
            if depth = (4, 5), 4 LSTMs will be added to the encoding side and
            5 LSTMs will be added to the decoding side.
    dropout : Dropout probability in between layers.
    '''

    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim
    encoder = RecurrentSequential(unroll=unroll, stateful=stateful)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[-1])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    decoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  decode=True, output_length=output_length)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], hidden_dim)))

    if depth[1] == 1:
        decoder.add(LSTMCell(output_dim))
    else:
        decoder.add(LSTMCell(hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMCell(hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMCell(output_dim))

    return encoder, decoder


def AttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
                     batch_size=None, input_shape=None, input_length=None,
                     input_dim=None, hidden_dim=None, depth=1,
                     bidirectional=True, unroll=False, stateful=False, dropout=0.0, ):
    '''
    This is an attention Seq2seq model based on [3].
    Here, there is a soft allignment between the input and output sequence elements.
    A bidirection encoder is used by default. There is no hidden state transfer in this
    model.
    The  math:
            Encoder:
            X = Input Sequence of length m.
            H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
            so H is a sequence of vectors of length m.
            Decoder:
    y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
    and v (called the context vector) is a weighted sum over H:
    v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)
    The weight alpha[i, j] for each hj is computed as follows:
    energy = a(s(i-1), H(j))
    alpha = softmax(energy)
    Where a is a feed forward network.
    '''

    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True

    encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  return_sequences=True)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum')
        encoder.forward_layer.build(shape)
        encoder.backward_layer.build(shape)
        # patch
        encoder.layer = encoder.forward_layer

    # encoded = encoder(_input)
    decoder = RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
    if depth[1] == 1:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    else:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))

    # inputs = [_input]
    # decoded = decoder(encoded)
    # model = Model(inputs, decoded)
    return encoder, decoder


def SingleCNNSeq2Seq(filter_num, kernel_size, strides,
               output_dim, output_length, hidden_dim=None, input_shape=None,
                  batch_size=None, batch_input_shape=None, input_dim=None,
                  input_length=None, depth=1, dropout=0.0, unroll=False,
                  stateful=False, model_type='simple'):
    input = Input(shape=input_shape)
    conv1Output = Conv2D(filters=filter_num, kernel_size=kernel_size,
                         strides=strides)(input)
    reshapeOutput = Reshape([conv1Output.get_shape().as_list()[1],
                             conv1Output.get_shape().as_list()[3]])(conv1Output)
    conv1Output = BatchNormalization()(reshapeOutput)
    conv1Output = Activation('relu')(conv1Output)
    # pool1Output = MaxPool2D((conv1Output.get_shape().as_list()[1], 1))(conv1Output)
    # reshapeOutput = Reshape([pool1Output.get_shape().as_list()[3], 1])(pool1Output)
    if model_type == 'attention':
        encoder, decoder = AttentionSeq2Seq(output_dim, output_length,
                                        batch_size=batch_size,
                                        input_shape=tuple(conv1Output.get_shape().as_list()[1:]),
                                        input_length=input_length, input_dim=input_dim,
                                        hidden_dim=hidden_dim, depth=depth, unroll=unroll,
                                        stateful=stateful, dropout=dropout)
    else:
        encoder, decoder = SimpleSeq2Seq(output_dim, output_length,
                                     hidden_dim, tuple(conv1Output.get_shape().as_list()[1:]),
                                     batch_size, batch_input_shape,
                                     input_dim, input_length,
                                     depth, dropout,
                                     unroll, stateful)
    seq2seqEncoderOutput = encoder(conv1Output)
    seq2seqDecoderOutput = decoder(seq2seqEncoderOutput)
    output = TimeDistributed(Dense(2, activation=softmax))(seq2seqDecoderOutput)
    return Model(input, output)


def DeepCNNSeq2Seq(filter_num, kernel_size, strides,
                     output_dim, output_length, label_size, hidden_dim=None, input_shape=None,
                     batch_size=None, batch_input_shape=None, input_dim=None,
                     input_length=None, depth=1, dropout=0.0, unroll=False,
                     stateful=False, model_type='simple'):
    input = Input(shape=input_shape)
    conv1Output = Conv2D(filters=filter_num, kernel_size=kernel_size,
                         strides=strides)(input)
    reshapeOutput = Reshape([conv1Output.get_shape().as_list()[1],
                             conv1Output.get_shape().as_list()[3]])(conv1Output)
    conv1Output = BatchNormalization()(reshapeOutput)
    conv1Output = Activation('relu')(conv1Output)
    conv1Output = Dropout(dropout)(conv1Output)
    conv1Output = Reshape([conv1Output.get_shape().as_list()[1],
                           conv1Output.get_shape().as_list()[2], 1])(conv1Output)
    conv2Output = Conv2D(filters=filter_num / 2, kernel_size=[3, filter_num],
                         strides=[2, 1])(conv1Output)
    conv2Output = Reshape([conv2Output.get_shape().as_list()[1],
                           conv2Output.get_shape().as_list()[3]])(conv2Output)
    conv2Output = BatchNormalization()(conv2Output)
    conv2Output = Activation('relu')(conv2Output)
    conv2Output = Dropout(dropout)(conv2Output)
    # pool1Output = MaxPool2D((conv1Output.get_shape().as_list()[1], 1))(conv1Output)
    # reshapeOutput = Reshape([pool1Output.get_shape().as_list()[3], 1])(pool1Output)
    if model_type == 'attention':
        encoder, decoder = AttentionSeq2Seq(output_dim, output_length,
                                            batch_size=batch_size,
                                            input_shape=tuple(conv2Output.get_shape().as_list()[1:]),
                                            input_length=input_length, input_dim=input_dim,
                                            hidden_dim=hidden_dim, depth=depth, unroll=unroll,
                                            stateful=stateful, dropout=dropout)
    else:
        encoder, decoder = SimpleSeq2Seq(output_dim, output_length,
                                         hidden_dim, tuple(conv2Output.get_shape().as_list()[1:]),
                                         batch_size, batch_input_shape,
                                         input_dim, input_length,
                                         depth, dropout,
                                         unroll, stateful)
    seq2seqEncoderOutput = encoder(conv2Output)
    seq2seqDecoderOutput = decoder(seq2seqEncoderOutput)
    output = TimeDistributed(Dense(label_size, activation=softmax))(seq2seqDecoderOutput)
    return Model(input, output)


def DeepCNN(input_shape, filter_num, kernel_size, strides, label_size, dropout=0.0):
    input = Input(shape=input_shape)
    conv1Output = Conv2D(filters=filter_num, kernel_size=kernel_size,
                         strides=strides)(input)
    reshapeOutput = Reshape([conv1Output.get_shape().as_list()[1],
                             conv1Output.get_shape().as_list()[3]])(conv1Output)
    conv1Output = BatchNormalization()(reshapeOutput)
    conv1Output = Activation('relu')(conv1Output)
    conv1Output = Dropout(dropout)(conv1Output)
    conv1Output = Reshape([conv1Output.get_shape().as_list()[1],
                           conv1Output.get_shape().as_list()[2], 1])(conv1Output)
    conv2Output = ZeroPadding2D((1, 0))(conv1Output)
    conv2Output = Conv2D(filters=filter_num / 2, kernel_size=[3, filter_num],
                         strides=[2, 1])(conv2Output)
    conv2Output = Reshape([conv2Output.get_shape().as_list()[1],
                           conv2Output.get_shape().as_list()[3]])(conv2Output)
    conv2Output = BatchNormalization()(conv2Output)
    conv2Output = Activation('relu')(conv2Output)
    conv2Output = Dropout(dropout)(conv2Output)
    conv2Output = TimeDistributed(Dense(16, activation=relu))(conv2Output)
    conv2Output = TimeDistributed(Dense(label_size, activation=softmax))(conv2Output)
    return Model(input, conv2Output)


def MixSignalDecoder(input_shape, filter_num, kernel_size, strides, label_size, signal_type_shape, dropout=0.0):
    input = Input(shape=input_shape)
    signal_type = Input(shape=signal_type_shape)
    conv1Output = Conv2D(filters=filter_num, kernel_size=kernel_size,
                         strides=strides)(input)
    reshapeOutput = Reshape([conv1Output.get_shape().as_list()[1],
                             conv1Output.get_shape().as_list()[3]])(conv1Output)
    conv1Output = BatchNormalization()(reshapeOutput)
    conv1Output = Activation('relu')(conv1Output)
    conv1Output = Dropout(dropout)(conv1Output)
    conv1Output = Concatenate(axis=-1)([conv1Output,
                                        RepeatVector(conv1Output.get_shape().as_list()[1])(signal_type)])
    conv1Output = Reshape([conv1Output.get_shape().as_list()[1],
                           conv1Output.get_shape().as_list()[2], 1])(conv1Output)
    conv2Output = ZeroPadding2D((1, 0))(conv1Output)
    conv2Output = Conv2D(filters=filter_num / 2, kernel_size=[3, conv1Output.get_shape().as_list()[2]],
                         strides=[2, 1])(conv2Output)
    conv2Output = Reshape([conv2Output.get_shape().as_list()[1],
                           conv2Output.get_shape().as_list()[3]])(conv2Output)
    conv2Output = BatchNormalization()(conv2Output)
    conv2Output = Activation('relu')(conv2Output)
    conv2Output = Dropout(dropout)(conv2Output)
    conv2Output = Concatenate(axis=-1)([conv2Output,
                                        RepeatVector(conv2Output.get_shape().as_list()[1])(signal_type)])
    conv2Output = TimeDistributed(Dense(16, activation=relu))(conv2Output)
    conv2Output = TimeDistributed(Dense(label_size, activation=softmax))(conv2Output)
    return Model([input, signal_type], conv2Output)


def DeeperCNN(input_shape, filter_num, kernel_size, strides, label_size, dropout=0.0):
    """
    Now is not available

    :param input_shape:
    :param filter_num:
    :param kernel_size:
    :param strides:
    :param label_size:
    :param dropout:
    :return:
    """
    input = Input(shape=input_shape)
    conv1Output = Conv2D(filters=filter_num, kernel_size=kernel_size,
                         strides=strides)(input)
    reshapeOutput = Reshape([conv1Output.get_shape().as_list()[1],
                             conv1Output.get_shape().as_list()[3]])(conv1Output)
    conv1Output = BatchNormalization()(reshapeOutput)
    conv1Output = Activation('relu')(conv1Output)
    conv1Output = Dropout(dropout)(conv1Output)
    conv1Output = Reshape([conv1Output.get_shape().as_list()[1],
                           conv1Output.get_shape().as_list()[2], 1])(conv1Output)
    conv2Output = Conv2D(filters=filter_num/2, kernel_size=[9, filter_num],
                         strides=[5, 1])(conv1Output)
    conv2Output = Reshape([conv2Output.get_shape().as_list()[1],
                           conv2Output.get_shape().as_list()[3]])(conv2Output)
    conv2Output = BatchNormalization()(conv2Output)
    conv2Output = Activation('relu')(conv2Output)
    conv2Output = Dropout(dropout)(conv2Output)
    conv2Output = Reshape([conv2Output.get_shape().as_list()[1],
                           conv2Output.get_shape().as_list()[2], 1])(conv2Output)
    conv3Output = ZeroPadding2D((1, 0))(conv2Output)
    conv3Output = Conv2D(filters=filter_num/4, kernel_size=[3, filter_num/2],
                         strides=[2, 1])(conv3Output)
    conv3Output = Reshape([conv3Output.get_shape().as_list()[1],
                           conv3Output.get_shape().as_list()[3]])(conv3Output)
    conv3Output = BatchNormalization()(conv3Output)
    conv3Output = Activation('relu')(conv3Output)
    conv3Output = Dropout(dropout)(conv3Output)
    conv3Output = TimeDistributed(Dense(filter_num/4, activation=relu))(conv3Output)
    conv3Output = TimeDistributed(Dense(label_size, activation=softmax))(conv3Output)
    return Model(input, conv3Output)





