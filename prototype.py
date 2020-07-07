from tensorflow import keras
from tensorflow.compat import v1 as _tf
import pickle
import numpy as np
import turicreate.toolkits._tf_utils as _utils


LSTM_H = 200
CONV_H = 64
prediction_window=50
input_shape=(1000, 6)
batch_size = 32
seed=123

model = keras.models.Sequential()

# XXX: label OHE?

inputs = keras.Input(shape=(1000,6))

dense = keras.layers.Conv1D(
    filters=CONV_H,
    kernel_size=(50),
    padding='same',
    strides=prediction_window,
    input_shape=input_shape,
    use_bias=True,
    activation='relu',
)

cur_outputs = dense(inputs)

dropout = keras.layers.Dropout(
    rate=0.2,
    seed=seed,
)
cur_outputs = dropout(cur_outputs)

lstm = keras.layers.LSTM(
    units=LSTM_H,
    return_sequences=True,
    use_bias=True,
)
cur_outputs = lstm(cur_outputs)

dense2 = keras.layers.Dense(128)
cur_outputs = dense2(cur_outputs)

batch_norm = keras.layers.BatchNormalization()
cur_outputs = batch_norm(cur_outputs)

relu = keras.layers.ReLU()
cur_outputs = relu(cur_outputs)

dropout = keras.layers.Dropout(
    rate=0.5,
    seed=seed,
)
cur_outputs = dropout(cur_outputs)

dense3 = keras.layers.Dense(6, use_bias=False)
cur_outputs = dense3(cur_outputs)

softmax = keras.layers.Softmax()
cur_outputs = softmax(cur_outputs)

model = keras.Model(inputs=inputs, outputs=cur_outputs)


with open('./net_params.pickle', 'rb') as f:
    net_params = pickle.load(f)


l = model.layers[1]
l.set_weights(
    (_utils.convert_conv1d_coreml_to_tf(net_params["conv_weight"]),
    net_params["conv_bias"])
    )



def get_params(order):
    i2h = []
    for i in order:
        i2h.append(eval('net_params["lstm_i2h_%s_weight"]' % i))
    i2h = np.concatenate(i2h, axis=0)

    h2h = []
    for i in order:
        h2h.append(eval('net_params["lstm_h2h_%s_weight"]' % i))
    h2h = np.concatenate(h2h, axis=0)

    bias = []
    for i in order:
        bias.append(eval('net_params["lstm_h2h_%s_bias"]' % i))
    bias = np.concatenate(bias, axis=0)

    i2h = np.swapaxes(i2h, 1, 0)
    h2h = np.swapaxes(h2h, 1, 0)
    return (i2h, h2h, bias)


l = model.layers[3]
l.set_weights(
    get_params(('i', 'f', 'c', 'o'))
)

l = model.layers[4]
l.set_weights(
    (
        net_params['dense0_weight'].reshape(128, 200).swapaxes(0, 1),
        net_params['dense0_bias']
    )
)

# Batch Norm
l = model.layers[5]
l.set_weights(
    (net_params['bn_gamma'],
     net_params['bn_beta'],
     net_params['bn_running_mean'],
     net_params['bn_running_var']
    )
)

#import ipdb; ipdb.set_trace()


l = model.layers[8]
l.set_weights(
    (
        net_params['dense1_weight'].reshape((6, 128)).swapaxes(0,1),
      )
)

np.random.seed(123)
x = np.random.rand(32, 1000, 6)

y = model.predict(x)
print(y)


print(y[10][5][1])   # 0.25368547

import tensorflow as tf

model.compile(
    loss=tf.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    sample_weight_mode="temporal"
)

with open('./feed_dict_train', 'br') as f:
    feed_dict = pickle.load(f)

loss = model.train_on_batch(
    x=feed_dict['input'],
    y=keras.utils.to_categorical(feed_dict['labels']),   # num_classes
    sample_weight=np.reshape(feed_dict['weights'], (32,20))
)
