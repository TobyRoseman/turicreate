from tensorflow import keras
from tensorflow.compat import v1 as _tf
import pickle
import numpy as np
import turicreate.toolkits._tf_utils as _utils


LSTM_H = 200
CONV_H = 64
prediction_window=50
input_shape=(1000, 6)
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


'''
# if self.is_training:
model.add(
    keras.layers.Dropout(
        rate=0.2,
        seed=seed,
    )
)


import tensorflow as tf
from tensorflow import keras
x = tf.random.normal([32, 20, 64])
'''

lstm = keras.layers.LSTM(
    units=LSTM_H,
    return_sequences=True,
    use_bias=True,

    #recurrent_initializer="zeros"
    
)

cur_outputs = lstm(cur_outputs)

model = keras.Model(inputs=inputs, outputs=cur_outputs)


'''
def custom_loss(y_true, y_pred):
    pass


optimizer = keras.optimizers.Adam(learning_rate=1e-3)
'''

'''
model.compile(
    loss=keras.losses.categorical_crossentropy,



)      
'''


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


from itertools import permutations
for cur_order in permutations(['i', 'c', 'f', 'o']):
    l = model.layers[2]
    l.set_weights(
        get_params(cur_order)
    )

    x = np.zeros((32, 1000, 6))
    y = model.predict(x)
    print(cur_order, np.sum(y[0]))

