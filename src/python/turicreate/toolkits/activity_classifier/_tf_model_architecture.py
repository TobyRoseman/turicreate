# -*- coding: utf-8 -*-
# Copyright © 2019 Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import turicreate.toolkits._tf_utils as _utils
from .._tf_model import TensorFlowModel

import numpy as _np

from turicreate._deps.minimal_package import _minimal_package_import_check


def _lazy_import_tensorflow():
    _tf = _minimal_package_import_check("tensorflow.compat.v1")

    # This toolkit is compatible with TensorFlow V2 behavior.
    # However, until all toolkits are compatible, we must call `disable_v2_behavior()`.
    _tf.disable_v2_behavior()

    _tf = _minimal_package_import_check("tensorflow")
    return _tf


# Constant parameters for the neural network
CONV_H = 64
LSTM_H = 200
DENSE_H = 128


class ActivityTensorFlowModel(TensorFlowModel):
    def __init__(
        self,
        net_params,
        batch_size,
        num_features,
        num_classes,
        prediction_window,
        seq_len,
        seed,
    ):
        _utils.suppress_tensorflow_warnings()

        self.num_classes = num_classes
        self.batch_size = batch_size

        tf = _lazy_import_tensorflow()
        keras = tf.keras

        #############################################
        # Define the Neural Network
        #############################################
        inputs = keras.Input(shape=(prediction_window * seq_len, num_features))

        # First dense layer
        dense = keras.layers.Conv1D(
            filters=CONV_H,
            kernel_size=(prediction_window),
            padding='same',
            strides=prediction_window,
            use_bias=True,
            activation='relu',
        )
        cur_outputs = dense(inputs)

        # First dropout layer
        dropout = keras.layers.Dropout(
            rate=0.2,
            seed=seed,
        )
        cur_outputs = dropout(cur_outputs)

        # LSTM layer
        lstm = keras.layers.LSTM(
            units=LSTM_H,
            return_sequences=True,
            use_bias=True,
        )
        cur_outputs = lstm(cur_outputs)

        # Second dense layer
        dense2 = keras.layers.Dense(DENSE_H)
        cur_outputs = dense2(cur_outputs)

        # Batch norm layer
        batch_norm = keras.layers.BatchNormalization()
        cur_outputs = batch_norm(cur_outputs)

        # ReLU layer
        relu = keras.layers.ReLU()
        cur_outputs = relu(cur_outputs)

        # Final dropout layer
        dropout = keras.layers.Dropout(rate=0.5, seed=seed)
        cur_outputs = dropout(cur_outputs)

        # Final dense layer
        dense3 = keras.layers.Dense(num_classes, use_bias=False)
        cur_outputs = dense3(cur_outputs)

        # Softmax layer
        softmax = keras.layers.Softmax()
        cur_outputs = softmax(cur_outputs)

        self.model = keras.Model(inputs=inputs, outputs=cur_outputs)
        self.model.compile(
            loss=tf.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            sample_weight_mode="temporal"
        )

        #############################################
        # Load the Weights of the Neural Network
        #############################################
        for key in net_params.keys():
            net_params[key] = _utils.convert_shared_float_array_to_numpy(net_params[key])

        # Set weight for first dense layer
        l = self.model.layers[1]
        l.set_weights(
            (_utils.convert_conv1d_coreml_to_tf(net_params["conv_weight"]),
             net_params["conv_bias"])
        )

        # Set LSTM weights
        i2h, h2h, bias = [], [], []
        for i in ('i', 'f', 'c', 'o'):
            i2h.append(eval('net_params["lstm_i2h_%s_weight"]' % i))
            h2h.append(eval('net_params["lstm_h2h_%s_weight"]' % i))
            bias.append(eval('net_params["lstm_h2h_%s_bias"]' % i))
        i2h = _np.concatenate(i2h, axis=0)
        h2h = _np.concatenate(h2h, axis=0)
        bias = _np.concatenate(bias, axis=0)
        i2h = _np.swapaxes(i2h, 1, 0)
        h2h = _np.swapaxes(h2h, 1, 0)
        l = self.model.layers[3]
        l.set_weights((i2h, h2h, bias))

        # Set weight for second dense layer
        l = self.model.layers[4]
        l.set_weights(
            (
                net_params['dense0_weight'].reshape(DENSE_H, 200).swapaxes(0, 1),
                net_params['dense0_bias']
            )
        )

        # Set batch Norm weights
        l = self.model.layers[5]
        l.set_weights(
            (
                net_params['bn_gamma'],
                net_params['bn_beta'],
                net_params['bn_running_mean'],
                net_params['bn_running_var']
            )
        )

        # Set weights for last dense layer
        l = self.model.layers[8]
        l.set_weights(
            (
                net_params['dense1_weight'].reshape((6, 128)).swapaxes(0,1),
            )
        )

    def train(self, feed_dict):
        """
        Run session for training with new batch of data (inputs, labels and weights)

        Parameters
        ----------
        feed_dict: Dictionary
            Dictionary to store a batch of input data, corresponding labels and weights. This is currently
            passed from the ac_data_iterator.cpp file when a new batch of data is sent.

        Returns
        -------
        result: Dictionary
            Loss per batch and probabilities
        """
        #import ipdb; ipdb.set_trace()

        for key in feed_dict.keys():
            feed_dict[key] = _utils.convert_shared_float_array_to_numpy(feed_dict[key])
            feed_dict[key] = _np.squeeze(feed_dict[key], axis=1)
            feed_dict[key] = _np.reshape(
                feed_dict[key],
                (
                    feed_dict[key].shape[0],
                    feed_dict[key].shape[1],
                    feed_dict[key].shape[2],
                ),
            )

        _, loss, probs = self.sess.run(
            [self.train_op, self.loss_per_seq, self.probs],
            feed_dict={
                self.data: feed_dict["input"],
                self.target: feed_dict["labels"],
                self.weight: feed_dict["weights"],
                self.is_training: True,
            },
        )

        prob = _np.array(probs)
        probabilities = _np.reshape(
            prob, (prob.shape[0], prob.shape[1] * prob.shape[2])
        )
        result = {"loss": _np.array(loss), "output": probabilities}
        return result

    def predict(self, feed_dict):
        """
        Run session for predicting with new batch of validation data (inputs, labels and weights) as well as test data (inputs)

        Parameters
        ----------
        feed_dict: Dictionary
            Dictionary to store a batch of input data, corresponding labels and weights. This is currently
            passed from the ac_data_iterator.cpp file when a new batch of data is sent.

        Returns
        -------
        result: Dictionary
            Loss per batch and probabilities (in case of validation data)
            Probabilities (in case only inputs are provided)
        """
        for key in feed_dict.keys():
            feed_dict[key] = _utils.convert_shared_float_array_to_numpy(feed_dict[key])
            feed_dict[key] = _np.squeeze(feed_dict[key], axis=1)
            feed_dict[key] = _np.reshape(
                feed_dict[key],
                (
                    feed_dict[key].shape[0],
                    feed_dict[key].shape[1],
                    feed_dict[key].shape[2],
                ),
            )

        if len(feed_dict.keys()) == 1:
            # Predict path
            prob = self.model.predict(feed_dict['input'])
            probabilities = _np.reshape(
                prob, (prob.shape[0], prob.shape[1] * prob.shape[2])
            )
            result = {"output": probabilities}
        else:
            loss, probs = self.sess.run(
                [self.loss_per_seq, self.probs],
                feed_dict={
                    self.data: feed_dict["input"],
                    self.target: feed_dict["labels"],
                    self.weight: feed_dict["weights"],
                    self.is_training: False,
                },
            )
            prob = _np.array(probs)
            probabilities = _np.reshape(
                prob, (prob.shape[0], prob.shape[1] * prob.shape[2])
            )
            result = {"loss": _np.array(loss), "output": probabilities}
        return result

    def export_weights(self):
        """
        Function to store TensorFlow weights back to into a dict in CoreML format to be used
        by the C++ implementation

        Returns
        -------
        tf_export_params: Dictionary
            Dictionary of weights from TensorFlow stored as {weight_name: weight_value}
        """
        _tf = _lazy_import_tensorflow()
        tf_export_params = {}
        with self.ac_graph.as_default():
            tvars = _tf.trainable_variables()
            tvars_vals = self.sess.run(tvars)

        #import ipdb; ipdb.set_trace()

        for var, val in zip(tvars, tvars_vals):
            if "weight" in var.name:
                if var.name.startswith("conv"):

                    tf_export_params[
                        var.name.split(":")[0]
                    ] = _utils.convert_conv1d_tf_to_coreml(val)
                elif var.name.startswith("dense"):
                    tf_export_params[
                        var.name.split(":")[0]
                    ] = _utils.convert_dense_tf_to_coreml(val)
            elif var.name.startswith("rnn/lstm_cell/kernel"):
                (
                    i2h_i,
                    i2h_c,
                    i2h_f,
                    i2h_o,
                    h2h_i,
                    h2h_c,
                    h2h_f,
                    h2h_o,
                ) = _utils.convert_lstm_weight_tf_to_coreml(val, CONV_H)
                tf_export_params["lstm_i2h_i_weight"] = i2h_i
                tf_export_params["lstm_i2h_c_weight"] = i2h_c
                tf_export_params["lstm_i2h_f_weight"] = i2h_f
                tf_export_params["lstm_i2h_o_weight"] = i2h_o
                tf_export_params["lstm_h2h_i_weight"] = h2h_i
                tf_export_params["lstm_h2h_c_weight"] = h2h_c
                tf_export_params["lstm_h2h_f_weight"] = h2h_f
                tf_export_params["lstm_h2h_o_weight"] = h2h_o
            elif var.name.startswith("rnn/lstm_cell/bias"):
                (
                    h2h_i_bias,
                    h2h_c_bias,
                    h2h_f_bias,
                    h2h_o_bias,
                ) = _utils.convert_lstm_bias_tf_to_coreml(val)
                tf_export_params["lstm_h2h_i_bias"] = h2h_i_bias
                tf_export_params["lstm_h2h_c_bias"] = h2h_c_bias
                tf_export_params["lstm_h2h_f_bias"] = h2h_f_bias
                tf_export_params["lstm_h2h_o_bias"] = h2h_o_bias
            elif var.name.startswith("batch_normalization"):
                tf_export_params["bn_" + var.name.split("/")[-1][0:-2]] = _np.array(val)
            else:
                tf_export_params[var.name.split(":")[0]] = _np.array(val)

        tvars = _tf.global_variables()
        tvars_vals = self.sess.run(tvars)
        for var, val in zip(tvars, tvars_vals):
            if "moving_mean" in var.name:
                tf_export_params["bn_running_mean"] = _np.array(val)
            if "moving_variance" in var.name:
                tf_export_params["bn_running_var"] = _np.array(val)
        for layer_name in tf_export_params.keys():
            tf_export_params[layer_name] = _np.ascontiguousarray(
                tf_export_params[layer_name]
            )
        return tf_export_params
