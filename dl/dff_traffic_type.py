'''
@date:11/10/2017
@author:AshrafAbdul
'''

import tflearn
import tensorflow as tf
import os
from utils.prepare_data import prepare_data

NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/test_cs.csv'



LOG_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/model/logs/'
CHECKPOINT_DIR  = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/model/checkpoint/'
BEST_CHECKPOINT_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/model/best_checkpoint/'



relu_weights_init = tflearn.initializations.xavier (seed=20171011)
relu_bias_init = tf.contrib.keras.initializers.Constant(value=0.001)
relu_regularizer = 'L2'
softmax_regularizer = 'L2'
softmax_weights_init = tflearn.initializations.xavier (seed=20171013)
softmax_bias_init = 'zeros'

input = tflearn.input_data(shape=[None, 121],name='input')

hl_1 = tflearn.fully_connected(input, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_1')
hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_2')
hl_3 = tflearn.fully_connected(hl_2, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_3')
hl_4 = tflearn.fully_connected(hl_3, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_4')
hl_5 = tflearn.fully_connected(hl_4, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_5')
hl_6 = tflearn.fully_connected(hl_5, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_6')

tt_output = tflearn.fully_connected(hl_6,2,activation='softmax',bias=True, weights_init=softmax_weights_init, regularizer=softmax_regularizer,  bias_init=softmax_bias_init,  name='tt_output')
network = tflearn.layers.estimator.regression (tt_output, loss='binary_crossentropy', learning_rate=0.0001)

model = tflearn.models.dnn.DNN (network, tensorboard_verbose=3, tensorboard_dir=LOG_DIR, checkpoint_path=CHECKPOINT_DIR, best_checkpoint_path=BEST_CHECKPOINT_DIR, best_val_accuracy=0.92)

# Run
previous_runs = os.listdir(LOG_DIR)
if len(previous_runs) == 0:
    run_number = 1
else:
    run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

run_id = 'run_%02d' % run_number

[[x_train,y_train],[x_val,y_val],[x_test,y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)
model.fit(x_train,y_train, n_epoch=50, shuffle=True, validation_set=(x_val, y_val),show_metric=True, batch_size=1024,snapshot_epoch=True,run_id=run_id)

test_accuracy = model.evaluate(x_test,y_test)
print("test_accuracy:",test_accuracy)