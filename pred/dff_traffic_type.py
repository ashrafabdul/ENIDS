'''
@date:14/10/2017
@author:AshrafAbdul
'''
import tflearn
import tensorflow as tf
from utils.prepare_data import prepare_data
import numpy as np

NSL_KDD_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/'
NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/test_cs.csv'

BEST_CHECKPOINT = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/model/best_checkpoint/9366'

input = tflearn.input_data(shape=[None, 121],name='input')
relu_weights_init = tflearn.initializations.xavier (seed=20171011)
relu_bias_init = tf.contrib.keras.initializers.Constant(value=0.001)
relu_regularizer = 'L2'
softmax_regularizer = 'L2'
softmax_weights_init = tflearn.initializations.xavier (seed=20171013)
softmax_bias_init = 'zeros'

# 3 layers overfit
hl_1 = tflearn.fully_connected(input, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_1')
hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_2')
hl_3 = tflearn.fully_connected(hl_2, 128, activation='relu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_3')

tt_output = tflearn.fully_connected(hl_3,2,activation='softmax',bias=True, weights_init=softmax_weights_init, regularizer=softmax_regularizer,  bias_init=softmax_bias_init,  name='tt_output')
network = tflearn.layers.estimator.regression (tt_output, loss='binary_crossentropy', learning_rate=0.0001)

model = tflearn.models.dnn.DNN (network)

model.load(BEST_CHECKPOINT)

[[x_train,y_train],[x_val,y_val],[x_test,y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)
x = np.concatenate((x_train,x_val,x_test),axis=0)
y = np.concatenate((y_train,y_val,y_test),axis=0)
test_accuracy = model.evaluate(x_test,y_test)
print("test_accuracy:",test_accuracy)
test_accuracy = model.evaluate(x,y)
print("full_accuracy:",test_accuracy)
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)
np.savetxt(NSL_KDD_DIR + "ac_train_pred.csv", y_train_pred, delimiter=",")
np.savetxt(NSL_KDD_DIR + "ac_val_pred.csv", y_val_pred, delimiter=",")
np.savetxt(NSL_KDD_DIR + "ac_test_pred.csv", y_test_pred, delimiter=",")