'''
@date:11/10/2017
@author:AshrafAbdul
'''

from tflearn.metrics import MTL_Accuracy
import tflearn
import tensorflow as tf
import os
from utils.prepare_data import prepare_data
import numpy as np

NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/master/oldsplit/train.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/master/oldsplit/val.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/master/oldsplit/test.csv'



LOG_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/logs/'
CHECKPOINT_DIR  = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/checkpoint/'
BEST_CHECKPOINT_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/best_checkpoint/mtl_tt_ac/'



def mtl_loss(y_pred, y_true):
    with tf.name_scope(None):
        begin_tt = [0,0]
        size_tt = [-1,2]

        begin_ac = [0,2]
        size_ac = [-1,5]

        begin_at = [0,7]
        size_at = [-1,40]

        y_pred_tt = tf.slice(y_pred,begin_tt,size_tt)
        y_true_tt = tf.slice(y_true,begin_tt,size_tt)

        y_pred_ac = tf.slice(y_pred,begin_ac,size_ac)
        y_true_ac = tf.slice(y_true,begin_ac,size_ac)

        #y_pred_at = tf.slice(y_pred,begin_at,size_at)
        #y_true_at = tf.slice(y_true,begin_at,size_at)

        loss_tt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tt, labels=y_true_tt))
        loss_ac = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_ac, labels=y_true_ac))
        #loss_at = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_at, labels=y_true_at))
        #loss_at = tf.scalar_mul(3,loss_at)
        loss = tf.add_n([loss_tt ,loss_ac])
        return loss


relu_weights_init = tflearn.initializations.xavier (seed=20171011)
relu_bias_init = tf.contrib.keras.initializers.Constant(value=0.001)
relu_regularizer = 'L2'
softmax_regularizer = 'L2'
softmax_weights_init = tflearn.initializations.xavier (seed=20171013)
softmax_bias_init = 'zeros'

input = tflearn.input_data(shape=[None, 121],name='input')

shared_hl_1 = tflearn.fully_connected(input, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=None, bias_init=relu_bias_init,   name='shared_hl_1')
shared_hl_2 = tflearn.fully_connected(shared_hl_1, 128,activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=None, bias_init=relu_bias_init,   name='shared_hl_2')
shared_hl_3 = tflearn.fully_connected(shared_hl_2,  128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=None, bias_init=relu_bias_init,   name='shared_hl_3')
shared_hl_4 = tflearn.fully_connected(shared_hl_3, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=None, bias_init=relu_bias_init,   name='shared_hl_4')
shared_hl_5 = tflearn.fully_connected(shared_hl_4, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=None, bias_init=relu_bias_init,   name='shared_hl_5')
shared_hl_6 = tflearn.fully_connected(shared_hl_5, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=None, bias_init=relu_bias_init,   name='shared_hl_6')

tt_hl_1 = tflearn.fully_connected(shared_hl_6, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer='L2', bias_init=relu_bias_init,   name='tt_hl_1')
ac_hl_1 = tflearn.fully_connected(shared_hl_6, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer='L2', bias_init=relu_bias_init,   name='ac_hl_1')



tt_output = tflearn.fully_connected(tt_hl_1,2,activation='softmax',bias=True, weights_init=softmax_weights_init, regularizer='L2',   bias_init=softmax_bias_init,  name='tt_output')
ac_output = tflearn.fully_connected(ac_hl_1,5,activation='softmax',bias=True, weights_init=softmax_weights_init, regularizer='L2',  bias_init=softmax_bias_init,  name='ac_output')

merge =tflearn.merge([tt_output,ac_output],mode='concat',name='merge')

#top3 = tflearn.metrics.Top_k(k=3)
mtl_acc = MTL_Accuracy()
network = tflearn.layers.estimator.regression (merge, loss=mtl_loss,metric=mtl_acc,  learning_rate=0.0001)
model = tflearn.models.dnn.DNN (network, tensorboard_verbose=3, tensorboard_dir=LOG_DIR, checkpoint_path=None, best_checkpoint_path=BEST_CHECKPOINT_DIR, best_val_accuracy=1.9)

# Run
previous_runs = os.listdir(LOG_DIR)
if len(previous_runs) == 0:
    run_number = 1
else:
    run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

run_id = 'run_%02d' % run_number

[[x_train,y_train],[x_val,y_val],[x_test,y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)
model.fit(x_train,y_train[:,0:7], n_epoch=200, shuffle=True, validation_set=(x_val, y_val[:,0:7]),show_metric=True, batch_size=512,snapshot_epoch=True,run_id=run_id)
