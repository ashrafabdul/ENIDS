'''
@date:14/10/2017
@author:AshrafAbdul
'''

import tflearn
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import PReLU


BEST_CHECKPOINT_TT = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/model/best_checkpoint/9446'
BEST_CHECKPOINT_AC = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/best_checkpoint/nwl/9898'
BEST_CHECKPOINT_AT = '/home/aabdul/projects/enids/data/NSL-KDD/attack_type/model/best_checkpoint/9522'
#BEST_CHECKPOINT_MTL = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/best_checkpoint/28894'
BEST_CHECKPOINT_MTL = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/best_checkpoint/mtl_tt_ac_at/29096'
BEST_CHECKPOINT_MTL_TT_AC = '/home/aabdul/projects/enids/data/NSL-KDD/master/model/best_checkpoint/mtl_tt_ac/19633'



def get_model_parameters(file_name, tensor_names):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    param_dict = {}
    for key in tensor_names:
        param_dict[key]=reader.get_tensor(key)
    return param_dict

def mtl_loss(y,y1):
    return None

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

    elif model_name == "mtl":

        input = tflearn.input_data(shape=[None, 121], name='input')

        shared_hl_1 = tflearn.fully_connected(input, 128, activation='prelu', bias=True, name='shared_hl_1')
        shared_hl_2 = tflearn.fully_connected(shared_hl_1, 128, activation='prelu', bias=True, name='shared_hl_2')
        shared_hl_3 = tflearn.fully_connected(shared_hl_2, 128, activation='prelu', bias=True, name='shared_hl_3')
        shared_hl_4 = tflearn.fully_connected(shared_hl_3, 128, activation='prelu', bias=True, name='shared_hl_4')
        shared_hl_5 = tflearn.fully_connected(shared_hl_4, 128, activation='prelu', bias=True, name='shared_hl_5')
        shared_hl_6 = tflearn.fully_connected(shared_hl_5, 128, activation='prelu', bias=True, name='shared_hl_6')

        tt_hl_1 = tflearn.fully_connected(shared_hl_6, 128, activation='prelu', bias=True, name='tt_hl_1')
        ac_hl_1 = tflearn.fully_connected(shared_hl_6, 128, activation='prelu', bias=True, name='ac_hl_1')
        at_hl_1 = tflearn.fully_connected(shared_hl_6, 128, activation='prelu', bias=True, name='at_hl_1')


        tt_output = tflearn.fully_connected(tt_hl_1, 2, activation='softmax', bias=True, name='tt_output')
        ac_output = tflearn.fully_connected(ac_hl_1, 5, activation='softmax', bias=True, name='ac_output')
        at_output = tflearn.fully_connected(at_hl_1, 40, activation='softmax', bias=True, name='at_output')

        merge = tflearn.merge([tt_output, ac_output, at_output], mode='concat', name='merge')
        network = tflearn.layers.estimator.regression(merge)
        model = tflearn.models.dnn.DNN(network)

        model.load(BEST_CHECKPOINT_MTL,weights_only=True)

    elif model_name == "mtl_tt_ac":

        input = tflearn.input_data(shape=[None, 121], name='input')

        shared_hl_1 = tflearn.fully_connected(input, 128, activation='prelu', bias=True, name='shared_hl_1')
        shared_hl_2 = tflearn.fully_connected(shared_hl_1, 128, activation='prelu', bias=True, name='shared_hl_2')
        shared_hl_3 = tflearn.fully_connected(shared_hl_2, 128, activation='prelu', bias=True, name='shared_hl_3')
        shared_hl_4 = tflearn.fully_connected(shared_hl_3, 128, activation='prelu', bias=True, name='shared_hl_4')
        shared_hl_5 = tflearn.fully_connected(shared_hl_4, 128, activation='prelu', bias=True, name='shared_hl_5')
        shared_hl_6 = tflearn.fully_connected(shared_hl_5, 128, activation='prelu', bias=True, name='shared_hl_6')

        tt_hl_1 = tflearn.fully_connected(shared_hl_6, 128, activation='prelu', bias=True, name='tt_hl_1')
        ac_hl_1 = tflearn.fully_connected(shared_hl_6, 128, activation='prelu', bias=True, name='ac_hl_1')

        tt_output = tflearn.fully_connected(tt_hl_1, 2, activation='softmax', bias=True, name='tt_output')
        ac_output = tflearn.fully_connected(ac_hl_1, 5, activation='softmax', bias=True, name='ac_output')

        merge = tflearn.merge([tt_output, ac_output,], mode='concat', name='merge')
        network = tflearn.layers.estimator.regression(merge)
        model = tflearn.models.dnn.DNN(network)

        model.load(BEST_CHECKPOINT_MTL_TT_AC,weights_only=True)

    return model



def get_best_model_keras(model_name):

    if model_name == "ac_hl6":

        tensor_names = ['hl_1/W', 'hl_1/b', 'hl_1/PReLU/alphas', 'hl_2/W', 'hl_2/b', 'hl_2/PReLU/alphas', 'hl_3/W', 'hl_3/b', 'hl_3/PReLU/alphas', 'hl_4/W', 'hl_4/b', 'hl_4/PReLU/alphas', 'hl_5/W','hl_5/b', 'hl_5/PReLU/alphas', 'hl_6/W', 'hl_6/b', 'hl_6/PReLU/alphas', 'ac_output/W', 'ac_output/b']
        param_dict = get_model_parameters(BEST_CHECKPOINT_AC,tensor_names)

        #PReLU(input_shape=[128],weights=param_dict['hl_2/PReLU/alphas'])
        model = Sequential()
        model.add(Dense(128, input_dim=121, activation='relu',use_bias=True,weights=[param_dict['hl_1/W'],param_dict['hl_1/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['hl_2/W'],param_dict['hl_2/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['hl_3/W'],param_dict['hl_3/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['hl_4/W'],param_dict['hl_4/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['hl_5/W'],param_dict['hl_5/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['hl_6/W'],param_dict['hl_6/b']]))
        model.add(Dense(5, activation='softmax', use_bias=True,weights=[param_dict['ac_output/W'],param_dict['ac_output/b']]))
        model.compile(loss='categorical_crossentropy',optimizer='adam')

        return model

    elif model_name == "mtl_ac":

        tensor_names = ['shared_hl_1/W', 'shared_hl_1/b', 'shared_hl_1/PReLU/alphas', 'shared_hl_2/W', 'shared_hl_2/b', 'shared_hl_2/PReLU/alphas', 'shared_hl_3/W', 'shared_hl_3/b', 'shared_hl_3/PReLU/alphas', 'shared_hl_4/W', 'shared_hl_4/b', 'shared_hl_4/PReLU/alphas', 'shared_hl_5/W','shared_hl_5/b', 'shared_hl_5/PReLU/alphas', 'shared_hl_6/W', 'shared_hl_6/b', 'shared_hl_6/PReLU/alphas', 'ac_output/W', 'ac_output/b','ac_hl_1/W', 'ac_hl_1/b']
        param_dict = get_model_parameters(BEST_CHECKPOINT_MTL,tensor_names)

        #PReLU(input_shape=[128],weights=param_dict['hl_2/PReLU/alphas'])
        model = Sequential()
        model.add(Dense(128, input_dim=121, activation='relu',use_bias=True,weights=[param_dict['shared_hl_1/W'],param_dict['shared_hl_1/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['shared_hl_2/W'],param_dict['shared_hl_2/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['shared_hl_3/W'],param_dict['shared_hl_3/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['shared_hl_4/W'],param_dict['shared_hl_4/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['shared_hl_5/W'],param_dict['shared_hl_5/b']]))
        model.add(Dense(128, activation='relu', use_bias=True,weights=[param_dict['shared_hl_6/W'],param_dict['shared_hl_6/b']]))
        model.add(Dense(128, activation='relu', use_bias=True, weights=[param_dict['ac_hl_1/W'], param_dict['ac_hl_1/b']]))
        model.add(Dense(5, activation='softmax', use_bias=True,weights=[param_dict['ac_output/W'],param_dict['ac_output/b']]))
        model.compile(loss='categorical_crossentropy',optimizer='adam')

        return model


if __name__=="__main__":
    model = get_best_model_keras('mtl')
    print(model.predict(np.array([[0]*121])))