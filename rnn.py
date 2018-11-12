###
# Copyright 2018 Edgard Chammas. All Rights Reserved.
# Licensed under the Creative Commons Attribution-NonCommercial International Public License, Version 4.0.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/legalcode
###

#!/usr/bin/python

import tensorflow as tf
import numpy as np
import math

from config import cfg
from util import LoadClasses
from cnn import FV
from cnn import NFeatures

Classes = LoadClasses(cfg.CHAR_LIST)

NClasses = len(Classes)

def RNN(Inputs, SeqLens, Scope):

    with tf.variable_scope(Scope):

        ################################################################
        #Construct batch sequences for LSTM

        maxLen = tf.reduce_max(SeqLens, 0)


        ndxs = tf.reshape(tf.range(0, SeqLens[0]), [SeqLens[0], 1])
        res = tf.gather_nd(Inputs, [ndxs])
        res = tf.reshape(res, [-1])
        zero_padding = tf.zeros([NFeatures * maxLen] - tf.shape(res), dtype=res.dtype)
        a_padded = tf.concat([res, zero_padding], 0)
        result = tf.reshape(a_padded, [maxLen, NFeatures])
        Inputs2 = result

        for n in range(1, cfg.BatchSize):
            offset = tf.cumsum(SeqLens)[n-1]
            ndxs = tf.reshape(tf.range(offset, SeqLens[n]+offset), [SeqLens[n], 1])
            res = tf.gather_nd(Inputs, [ndxs])
            res = tf.reshape(res, [-1])
            zero_padding = tf.zeros([NFeatures * maxLen] - tf.shape(res), dtype=res.dtype)
            a_padded = tf.concat([res, zero_padding], 0)
            result = tf.reshape(a_padded, [maxLen, NFeatures])
            Inputs2 = tf.concat([Inputs2, result], 0)


        ndxs = tf.reshape(tf.range(0, cfg.BatchSize * maxLen, maxLen), [cfg.BatchSize, 1])
        Inputs = tf.gather_nd(Inputs2, [ndxs])

        i = tf.constant(1)

        def condition(i, prev):
            minI = tf.less(i,maxLen)
            return minI

        def body(i, prev):
            ndxs = tf.reshape(tf.range(i, cfg.BatchSize * maxLen, maxLen), [cfg.BatchSize, 1])
            result = tf.gather_nd(Inputs2, [ndxs])
            next = tf.concat([prev, result], 0)
            addresult = tf.add(i,1)
            return [addresult, next]

        tensorShape = tf.TensorShape([None, cfg.BatchSize, NFeatures])
        i, Inputs = tf.while_loop(condition, body, [i, Inputs], shape_invariants=[i.get_shape(), tensorShape])

        ###############################################################
        #Construct LSTM layers

        initializer = tf.contrib.layers.xavier_initializer()

        stacked_rnn_forward = []
        for i in range(cfg.NLayers):
            lstmcell = tf.nn.rnn_cell.LSTMCell(num_units=cfg.NUnits, initializer=initializer, use_peepholes=True, state_is_tuple=True)
            stacked_rnn_forward.append(lstmcell)
        forward = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn_forward, state_is_tuple=True)

        stacked_rnn_backward = []
        for i in range(cfg.NLayers):
            lstmcell2 = tf.nn.rnn_cell.LSTMCell(num_units=cfg.NUnits, initializer=initializer, use_peepholes=True, state_is_tuple=True)
            stacked_rnn_backward.append(lstmcell2)
        backward = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn_backward, state_is_tuple=True)

        sequence_length = tf.cast(SeqLens, tf.int64)
        [fw_out, bw_out], _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward, cell_bw=backward, inputs=Inputs, time_major=True, dtype=tf.float32,sequence_length=sequence_length)

        # Batch normalize forward output
        mew,var_ = tf.nn.moments(fw_out,axes=[0])
        fw_out = tf.nn.batch_normalization(fw_out, mew, var_, 0.1, 1, 1e-6)

        # Batch normalize backward output
        mew,var_ = tf.nn.moments(bw_out,axes=[0])
        bw_out = tf.nn.batch_normalization(bw_out, mew, var_, 0.1, 1, 1e-6)

        # Reshaping forward, and backward outputs for affine transformation
        fw_out = tf.reshape(fw_out,[-1, cfg.NUnits])
        bw_out = tf.reshape(bw_out,[-1, cfg.NUnits])

        # Linear Layer params
        W_fw = tf.Variable(tf.truncated_normal(shape=[cfg.NUnits, NClasses], stddev=np.sqrt(2.0 / cfg.NUnits), dtype=tf.float32), dtype=tf.float32)
        W_bw = tf.Variable(tf.truncated_normal(shape=[cfg.NUnits, NClasses], stddev=np.sqrt(2.0 / cfg.NUnits), dtype=tf.float32), dtype=tf.float32)
        b_out = tf.constant(0.1,shape=[NClasses], dtype=tf.float32)

        # Perform an affine transformation
        allW = tf.matmul(fw_out,W_fw)
        allb = tf.matmul(bw_out,W_bw)
        bw = tf.add( allW, allb )
        logits =  tf.add(bw,  b_out)

        rnn_res = tf.reshape(logits, [-1, cfg.BatchSize, NClasses])
        return rnn_res

