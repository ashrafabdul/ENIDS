'''
@date:13/10/2017
@author:AshrafAbdul
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn


NSL_KDD_MASTER_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/master/train.csv'
NSL_KDD_MASTER_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/master/val.csv'
NSL_KDD_MASTER_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/master/test.csv'

INDEX_COL = ['data_index']
TRAFFIC_TYPE_COLS = ['traffic_type_anomalous','traffic_type_normal']
ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']
ATTACK_TYPE_COLS = ['attack_type_apache2','attack_type_back','attack_type_buffer_overflow','attack_type_ftp_write','attack_type_guess_passwd','attack_type_httptunnel','attack_type_imap','attack_type_ipsweep','attack_type_land','attack_type_loadmodule','attack_type_mailbomb','attack_type_mscan','attack_type_multihop','attack_type_named','attack_type_neptune','attack_type_nmap','attack_type_normal','attack_type_perl','attack_type_phf','attack_type_pod','attack_type_portsweep','attack_type_processtable','attack_type_ps','attack_type_rootkit','attack_type_saint','attack_type_satan','attack_type_sendmail','attack_type_smurf','attack_type_snmpgetattack','attack_type_snmpguess','attack_type_spy','attack_type_sqlattack','attack_type_teardrop','attack_type_udpstorm','attack_type_warezclient','attack_type_warezmaster','attack_type_worm','attack_type_xlock','attack_type_xsnoop','attack_type_xterm']

LOG_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/logs/'
CHECKPOINT_DIR  = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/checkpoint/'
BEST_CHECKPOINT_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/best_checkpoint/auc/'


def prepare_data(standardize_training_data=False):

    PRED_COLS = TRAFFIC_TYPE_COLS + ATTACK_CATEGORY_COLS + ATTACK_TYPE_COLS
    df_train = pd.read_csv(NSL_KDD_MASTER_TRAIN, dtype=np.float64)
    x_train = df_train[df_train.columns.difference(INDEX_COL + PRED_COLS)]
    y_train = df_train[PRED_COLS]
    print(y_train.head(2))
    if(standardize_training_data):
        for col in x_train.columns:
            std = x_train[col].std(ddof=0)
            mean = x_train[col].mean()
            if not std == 0:
                x_train[col] = (x_train[col] - mean)/ std
            else:
                print(col,std,mean)
    x_train = x_train.values
    y_train = y_train.values

    df_val = pd.read_csv(NSL_KDD_MASTER_VAL)
    x_val = df_val[df_val.columns.difference(INDEX_COL + PRED_COLS)]
    y_val = df_val[PRED_COLS]
    x_val = x_val.values
    y_val = y_val.values

    df_test = pd.read_csv(NSL_KDD_MASTER_TEST)
    x_test = df_test[df_test.columns.difference(INDEX_COL + PRED_COLS)]
    y_test = df_test[PRED_COLS]
    x_test = x_test.values
    y_test = y_test.values

    return [[x_train,y_train],[x_val,y_val],[x_test,y_test]]


[[x_train,y_train],[x_val,y_val],[x_test,y_test]] = prepare_data()

weights_init = tf.contrib.keras.initializers.he_uniform(seed=20171013)
bias_init = tf.contrib.keras.initializers.Constant(value=0.01)

input = tflearn.input_data(shape=[None, 121],name='input')
#shared_hl_1 = tflearn.fully_connected(input, 256, activation='relu', bias=True, weights_init=weights_init, regularizer=None, bias_init=bias_init,   name='shared_hl_1')
shared_hl_2 = tflearn.fully_connected(input, 128, activation='relu', bias=True, weights_init=weights_init, regularizer=None, bias_init=bias_init,   name='shared_hl_2')
shared_hl_3 = tflearn.fully_connected(shared_hl_2, 64, activation='relu', bias=True, weights_init=weights_init, regularizer=None, bias_init=bias_init,   name='shared_hl_3')

tt_hl_1 = tflearn.fully_connected(shared_hl_3, 128, activation='relu', bias=True, weights_init=weights_init, regularizer='L2', bias_init=bias_init,   name='tt_hl_1')
#tt_hl_2 = tflearn.fully_connected(tt_hl_1, 64, activation='relu', bias=True, weights_init=weights_init, regularizer='L1', bias_init=bias_init,   name='tt_hl_2')
ac_hl_1 = tflearn.fully_connected(shared_hl_3, 128, activation='relu', bias=True, weights_init=weights_init, regularizer='L2', bias_init=bias_init,   name='ac_hl_1')
#ac_hl_2 = tflearn.fully_connected(ac_hl_1, 64, activation='relu', bias=True, weights_init=weights_init, regularizer='L1', bias_init=bias_init,   name='ac_hl_2')
at_hl_1 = tflearn.fully_connected(shared_hl_3, 128, activation='relu', bias=True, weights_init=weights_init, regularizer='L2', bias_init=bias_init,   name='at_hl_1')
#at_hl_2 = tflearn.fully_connected(at_hl_1, 64, activation='relu', bias=True, weights_init=weights_init, regularizer='L', bias_init=bias_init,   name='at_hl_2')

weights_init = tflearn.initializations.xavier (seed=20171013)
tt_output = tflearn.fully_connected(tt_hl_1,2,activation='softmax',bias=True, weights_init=weights_init,  bias_init='zeros',  name='tt_output')
ac_output = tflearn.fully_connected(ac_hl_1,5,activation='softmax',bias=True, weights_init=weights_init,  bias_init='zeros',  name='ac_output')
at_output = tflearn.fully_connected(at_hl_1,40,activation='softmax',bias=True, weights_init=weights_init,  bias_init='zeros',  name='at_output')

network =tflearn.merge([tt_output,ac_output,at_output],mode='concat')

network = tflearn.layers.estimator.regression (network, loss='roc_auc_score', learning_rate=0.0001)


model = tflearn.models.dnn.DNN (network, tensorboard_verbose=3, tensorboard_dir=LOG_DIR, checkpoint_path=CHECKPOINT_DIR, best_checkpoint_path=BEST_CHECKPOINT_DIR, best_val_accuracy=0.85)
model.fit(x_train,y_train, n_epoch=1, shuffle=True, validation_set=(x_val, y_val),show_metric=True, batch_size=128,snapshot_epoch=True,run_id='mtl_auc')

test_accuracy = model.evaluate(x_test,y_test)
print("test_accuracy:",test_accuracy)