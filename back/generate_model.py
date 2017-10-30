'''
@date:11/10/2017
@author:AshrafAbdul
'''

import tflearn

layers = []
layer = [[si]]
def generate_model(input_size,layers)

    input = tflearn.input_data(shape=[None, +input_size+], name='input')
    num_layers = len(layers)
    for i in len(layers):

        hl_+i+ = tflearn.fully_connected(input, 125, activation='relu', bias=True, weights_init='truncated_normal', bias_init='zeros', name='hl+i+')
    hl_2 = tflearn.fully_connected(hl_1, 64, activation='relu', bias=True, weights_init='truncated_normal', bias_init='zeros', name='hl2')
    hl_3 = tflearn.fully_connected(hl_2, 32, activation='relu', bias=True, weights_init='truncated_normal', bias_init='zeros', name='hl3')
    hl_4 = tflearn.fully_connected(hl_3, 16, activation='relu', bias=True, weights_init='truncated_normal', bias_init='zeros', name='hl4')
    output = tflearn.fully_connected(hl_4, 2, activation='softmax', bias=True, weights_init='truncated_normal', bias_init='zeros', name='output')