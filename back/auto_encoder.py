'''
@date:13/10/2017
@author:AshrafAbdul
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
import os


NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/train.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/val.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/test.csv'

INDEX_COL = ['data_index']
TRAFFIC_TYPE_COLS = []
ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']
ATTACK_TYPE_COLS = []

LOG_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/logs/'
CHECKPOINT_DIR  = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/checkpoint/'
BEST_CHECKPOINT_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/attack_category/best_checkpoint/'


def prepare_data(standardize_training_data=False,scale=True):

    PRED_COLS = TRAFFIC_TYPE_COLS + ATTACK_CATEGORY_COLS + ATTACK_TYPE_COLS
    df_train = pd.read_csv(NSL_KDD_TRAIN, dtype=np.float32)
    x_train = df_train[df_train.columns.difference(INDEX_COL + PRED_COLS)]
    y_train = df_train[PRED_COLS]
    print(y_train.head(2))
    if(standardize_training_data):
        if(scale):
            for col in x_train.columns:
                min = x_train[col].min()
                max = x_train[col].max()
                print(col,max,min)
                if max - min == 0:
                    if not max == 0:
                        x_train[col] = x_train[col]/max
                else:
                    x_train[col] = (x_train[col] - min) / max - min
        else:
            for col in x_train.columns:
                std = x_train[col].std(ddof=0)
                mean = x_train[col].mean()
                if not std == 0:
                    x_train[col] = (x_train[col] - mean)/ std
                else:
                    print(col,std,mean)
    print(x_train.loc[:, x_train.isnull().any()])
    x_train = x_train.values
    y_train = y_train.values

    df_val = pd.read_csv(NSL_KDD_VAL)
    x_val = df_val[df_val.columns.difference(INDEX_COL + PRED_COLS)]
    y_val = df_val[PRED_COLS]
    x_val = x_val.values
    y_val = y_val.values

    df_test = pd.read_csv(NSL_KDD_TEST)
    x_test = df_test[df_test.columns.difference(INDEX_COL + PRED_COLS)]
    y_test = df_test[PRED_COLS]
    x_test = x_test.values
    y_test = y_test.values

    return [[x_train,y_train],[x_val,y_val],[x_test,y_test]]


[[x_train,y_train],[x_val,y_val],[x_test,y_test]] = prepare_data(standardize_training_data=True,scale=True)



previous_runs = os.listdir(LOG_DIR)
if len(previous_runs) == 0:
    run_number = 1
else:
    run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

run_id = 'run_%02d' % run_number

weights_init = tf.contrib.keras.initializers.he_uniform(seed=20171013)
bias_init = tf.contrib.keras.initializers.Constant(value=0.01)

input = tflearn.input_data(shape=[None, 121],name='input')
regularizer = None
hl_1 = tflearn.fully_connected(input, 256, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_1')
hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_2')
hl_3 = tflearn.fully_connected(hl_2, 64, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_3')
hl_5 = tflearn.fully_connected(hl_3, 121, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_5')

network = tflearn.layers.estimator.regression (hl_5, loss='mean_square', learning_rate=0.0001)

model = tflearn.models.dnn.DNN (network, clip_gradients=10.0, tensorboard_verbose=3, tensorboard_dir=LOG_DIR, checkpoint_path=CHECKPOINT_DIR, best_checkpoint_path=BEST_CHECKPOINT_DIR, best_val_accuracy=92.0)
model.fit(x_train,y_train, n_epoch=20, shuffle=True, validation_set=(x_val, y_val),show_metric=True, batch_size=1024,snapshot_epoch=True,run_id=run_id)

test_accuracy = model.evaluate(x_test,y_test)
print("test_accuracy:",test_accuracy)