'''
@date:13/10/2017
@author:AshrafAbdul
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
import os


NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/test_cs.csv'

INDEX_COL = ['data_index']
TRAFFIC_TYPE_COLS = []
ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']
ATTACK_TYPE_COLS = []

LOG_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/logs/'
CHECKPOINT_DIR  = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/checkpoint/'
BEST_CHECKPOINT_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/attack_category/best_checkpoint/'


def prepare_data(standardize_training_data=False):

    PRED_COLS = TRAFFIC_TYPE_COLS + ATTACK_CATEGORY_COLS + ATTACK_TYPE_COLS
    df_train = pd.read_csv(NSL_KDD_TRAIN, dtype=np.float64)
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



def generate_hidden_layers(num_layers,nodes,regularizer,relu_weights_init,relu_bias_init,softmax_weights_init, softmax_bias_init,softmax_size):

    network = "input = tflearn.input_data(shape=[None, 121],name='input')" + "\r\n"
    for i in range(1,num_layers+1):
        if(i==1):
            network = network + "hl_"+str(i)+" = tflearn.fully_connected(input, "+str(nodes[i-1])+", activation='relu', bias=True, weights_init="+relu_weights_init+", regularizer="+regularizer+", bias_init="+relu_bias_init+",   name='hl_"+str(i)+"')" + "\r\n"
        else:
            network = network + "hl_" + str(i) + " = tflearn.fully_connected(hl_"+str(i-1)+", " + str(nodes[i-1]) + ", activation='relu', bias=True, weights_init=" + relu_weights_init + ", regularizer=" + regularizer + ", bias_init=" + relu_bias_init + ",   name='hl_" + str(i) + "')" + "\r\n"
    network = network + "output = tflearn.fully_connected(hl_"+str(i)+", "+str(softmax_size)+", activation='softmax', bias=True, weights_init="+softmax_weights_init+", bias_init="+softmax_bias_init+", name='softmax_output')" + "\r\n"

    return network

def train_model(run_id):

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
hl_1 = tflearn.fully_connected(input, 128, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_1')
hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_2')
hl_3 = tflearn.fully_connected(hl_2, 128, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_3')
hl_4 = tflearn.fully_connected(hl_3, 128, activation='relu', bias=True, weights_init=weights_init, regularizer=regularizer, bias_init=bias_init,   name='hl_4')

weights_init = tflearn.initializations.xavier (seed=20171013)
ac_output = tflearn.fully_connected(hl_4,5,activation='softmax',bias=True, weights_init=weights_init,  bias_init='zeros',  name='ac_output')
network = tflearn.layers.estimator.regression (ac_output, loss='categorical_crossentropy', learning_rate=0.0001)

model = tflearn.models.dnn.DNN (network, clip_gradients=10.0, tensorboard_verbose=3, tensorboard_dir=LOG_DIR, checkpoint_path=CHECKPOINT_DIR, best_checkpoint_path=BEST_CHECKPOINT_DIR, best_val_accuracy=92.0)
model.fit(x_train,y_train, n_epoch=50, shuffle=True, validation_set=(x_val, y_val),show_metric=True, batch_size=8,snapshot_epoch=True,run_id=run_id)

test_accuracy = model.evaluate(x_test,y_test)
print("test_accuracy:",test_accuracy)

if __name__=="__main__":
    layers = [2,3,4,5,6]
    nodes = [[[128,128,128,16,8]]]



    layers = 3
    nodes = [128,64,32,16,8]
    regularizer = "'L2'"
    relu_weights_init = "tf.contrib.keras.initializers.he_uniform(seed=20171013)"
    relu_bias_init = "tf.contrib.keras.initializers.Constant(value=0.01)"
    softmax_weights_init = "tflearn.initializations.xavier (seed=20171013)"
    softmax_bias_init = "'zeros'"
    softmax_size = 5




    print(generate_hidden_layers(layers,nodes,regularizer,relu_weights_init,relu_bias_init,softmax_weights_init, softmax_bias_init,softmax_size))