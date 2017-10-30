'''
@date:14/10/2017
@author:AshrafAbdul
'''

import tflearn
import tensorflow as tf


BEST_CHECKPOINT_TT = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/model/best_checkpoint/9446'
BEST_CHECKPOINT_AC = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/best_checkpoint/9740'
BEST_CHECKPOINT_AT = '/home/aabdul/projects/enids/data/NSL-KDD/attack_type/model/best_checkpoint/9522'


def get_best_model(model_name):
    tf.reset_default_graph()
    model = None

    if model_name == "tt_hl6":

        # Traffic Type Model
        input = tflearn.input_data(shape=[None, 121], name='input')

        hl_1 = tflearn.fully_connected(input, 128, activation='relu', bias=True, name='hl_1')
        hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, name='hl_2')
        hl_3 = tflearn.fully_connected(hl_2, 128, activation='relu', bias=True, name='hl_3')
        hl_4 = tflearn.fully_connected(hl_3, 128, activation='relu', bias=True, name='hl_4')
        hl_5 = tflearn.fully_connected(hl_4, 128, activation='relu', bias=True, name='hl_5')
        hl_6 = tflearn.fully_connected(hl_5, 128, activation='relu', bias=True, name='hl_6')

        tt_output = tflearn.fully_connected(hl_6, 2, activation='softmax', bias=True,  name='tt_output')
        network = tflearn.layers.estimator.regression(tt_output, loss='binary_crossentropy')
        model = tflearn.models.dnn.DNN(network)

        model.load(BEST_CHECKPOINT_TT)
    elif model_name == "ac_hl6":

        # Attack Category Model
        input = tflearn.input_data(shape=[None, 121], name='input')

        hl_1 = tflearn.fully_connected(input, 128, activation='relu', bias=True, name='hl_1')
        hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, name='hl_2')
        hl_3 = tflearn.fully_connected(hl_2, 128, activation='relu', bias=True, name='hl_3')
        hl_4 = tflearn.fully_connected(hl_3, 128, activation='relu', bias=True, name='hl_4')
        hl_5 = tflearn.fully_connected(hl_4, 128, activation='relu', bias=True, name='hl_5')
        hl_6 = tflearn.fully_connected(hl_5, 128, activation='relu', bias=True, name='hl_6')

        ac_output = tflearn.fully_connected(hl_6, 5, activation='softmax', bias=True,  name='ac_output')
        network = tflearn.layers.estimator.regression(ac_output, loss='categorical_crossentropy')
        model = tflearn.models.dnn.DNN(network)

        model.load(BEST_CHECKPOINT_AC)
    elif model_name == "at_hl6":

        # Attack Type Model
        input = tflearn.input_data(shape=[None, 121], name='input')

        hl_1 = tflearn.fully_connected(input, 128, activation='relu', bias=True, name='hl_1')
        hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, name='hl_2')
        hl_3 = tflearn.fully_connected(hl_2, 128, activation='relu', bias=True, name='hl_3')
        hl_4 = tflearn.fully_connected(hl_3, 128, activation='relu', bias=True, name='hl_4')
        hl_5 = tflearn.fully_connected(hl_4, 128, activation='relu', bias=True, name='hl_5')
        hl_6 = tflearn.fully_connected(hl_5, 128, activation='relu', bias=True, name='hl_6')

        at_output = tflearn.fully_connected(hl_6, 40, activation='softmax', bias=True, name='at_output')
        network = tflearn.layers.estimator.regression(at_output, loss='categorical_crossentropy',)
        model = tflearn.models.dnn.DNN(network)

        model.load(BEST_CHECKPOINT_AT)

    return model




