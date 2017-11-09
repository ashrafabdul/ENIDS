'''
@date:11/10/2017
@author:AshrafAbdul
'''

import tflearn
import tensorflow as tf
import os
from utils.prepare_data import prepare_data
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/test_cs.csv'



LOG_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/logs/'
CHECKPOINT_DIR  = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/checkpoint/'
BEST_CHECKPOINT_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/best_checkpoint/nwl/'

ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']


def weighted_loss(y_pred, y_true):
    with tf.name_scope('weighted__loss'):


        begin_c1 = [0, 0]
        size_c1 = [-1, 1]

        begin_c2 = [0, 1]
        size_c2 = [-1, 1]

        begin_c3 = [0, 2]
        size_c3 = [-1, 1]

        begin_c4 = [0, 3]
        size_c4 = [-1, 1]

        begin_c5 = [0, 4]
        size_c5 = [-1, 1]

        y_pred_c1 = tf.slice(y_pred, begin_c1, size_c1)
        y_true_c1 = tf.slice(y_true, begin_c1, size_c1)

        y_pred_c2 = tf.slice(y_pred, begin_c2, size_c2)
        y_true_c2 = tf.slice(y_true, begin_c2, size_c2)

        y_pred_c3 = tf.slice(y_pred, begin_c3, size_c3)
        y_true_c3 = tf.slice(y_true, begin_c3, size_c3)

        y_pred_c4 = tf.slice(y_pred, begin_c4, size_c4)
        y_true_c4 = tf.slice(y_true, begin_c4, size_c4)

        y_pred_c5 = tf.slice(y_pred, begin_c5, size_c5)
        y_true_c5 = tf.slice(y_true, begin_c5, size_c5)

        y_pred_nc1 = tf.add_n([y_pred_c2,y_pred_c3,y_pred_c4,y_pred_c5])
        y_true_nc1 = tf.add_n([y_true_c2, y_true_c3, y_true_c4, y_true_c5])

        y_pred_nc2 = tf.add_n([y_pred_c1, y_pred_c3, y_pred_c4, y_pred_c5])
        y_true_nc2 = tf.add_n([y_true_c1, y_true_c3, y_true_c4, y_true_c5])

        y_pred_nc3 = tf.add_n([y_pred_c2, y_pred_c1, y_pred_c4, y_pred_c5])
        y_true_nc3 = tf.add_n([y_true_c2, y_true_c1, y_true_c4, y_true_c5])

        y_pred_nc4 = tf.add_n([y_pred_c2, y_pred_c3, y_pred_c1, y_pred_c5])
        y_true_nc4 = tf.add_n([y_true_c2, y_true_c3, y_true_c1, y_true_c5])

        y_pred_nc5 = tf.add_n([y_pred_c2, y_pred_c3, y_pred_c4, y_pred_c1])
        y_true_nc5 = tf.add_n([y_true_c2, y_true_c3, y_true_c4, y_true_c1])

        loss_c1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.concat([y_pred_c1, y_pred_nc1], 1),labels=tf.concat([y_true_c1, y_true_nc1], 1)))
        loss_c1 = tf.scalar_mul(1.44,loss_c1)
        loss_c2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.concat([y_pred_c2, y_pred_nc2], 1),labels=tf.concat([y_true_c2, y_true_nc2], 1)))

        loss_c3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.concat([y_pred_c3, y_pred_nc3], 1),labels=tf.concat([y_true_c3, y_true_nc3], 1)))
        loss_c3 = tf.scalar_mul(5.74,loss_c3)
        loss_c4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.concat([y_pred_c4, y_pred_nc4], 1),labels=tf.concat([y_true_c4, y_true_nc4], 1)))
        loss_c4 = tf.scalar_mul(19.85, loss_c4)
        loss_c5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.concat([y_pred_c5, y_pred_nc5], 1),labels=tf.concat([y_true_c5, y_true_nc5], 1)))
        loss_c5 = tf.scalar_mul(642, loss_c5)
        loss = tf.add_n([loss_c1,loss_c2,loss_c3,loss_c4,loss_c5])

        return loss


relu_weights_init = tflearn.initializations.xavier (seed=20171011)
relu_bias_init = tf.contrib.keras.initializers.Constant(value=0.001)
relu_regularizer = None
softmax_regularizer = 'L2'
softmax_weights_init = tflearn.initializations.xavier (seed=20171013)
softmax_bias_init = 'zeros'

input = tflearn.input_data(shape=[None, 121],name='input')

# 3 layers overfit
hl_1 = tflearn.fully_connected(input, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_1')
hl_2 = tflearn.fully_connected(hl_1, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_2')
hl_3 = tflearn.fully_connected(hl_2, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_3')
hl_4 = tflearn.fully_connected(hl_3, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_4')
hl_5 = tflearn.fully_connected(hl_4, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_5')
hl_6 = tflearn.fully_connected(hl_5, 128, activation='prelu', bias=True, weights_init=relu_weights_init, regularizer=relu_regularizer, bias_init=relu_bias_init,   name='hl_6')

ac_output = tflearn.fully_connected(hl_6,5,activation='softmax',bias=True, weights_init=softmax_weights_init, regularizer=softmax_regularizer,  bias_init=softmax_bias_init,  name='ac_output')


network = tflearn.layers.estimator.regression (ac_output, loss='categorical_crossentropy', learning_rate=0.0001)

model = tflearn.models.dnn.DNN (network, tensorboard_verbose=3, tensorboard_dir=LOG_DIR, checkpoint_path=None, best_checkpoint_path=BEST_CHECKPOINT_DIR, best_val_accuracy=0.97)

# Run
previous_runs = os.listdir(LOG_DIR)
if len(previous_runs) == 0:
    run_number = 1
else:
    run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

run_id = 'run_%02d' % run_number

[[x_train,y_train],[x_val,y_val],[x_test,y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)
model.fit(x_train,y_train, n_epoch=200, shuffle=True, validation_set=(x_val, y_val),show_metric=True, batch_size=1024,snapshot_epoch=True,run_id=run_id)

test_accuracy = model.evaluate(x_test,y_test)
print("test_accuracy:",test_accuracy)

y_test_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_test_pred = np.argmax(y_test_pred, axis=1)
print(classification_report(y_test,y_test_pred ,target_names=ATTACK_CATEGORY_COLS))
print(confusion_matrix(y_test,y_test_pred))