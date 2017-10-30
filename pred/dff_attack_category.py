'''
@date:14/10/2017
@author:AshrafAbdul
'''
import tflearn
import tensorflow as tf
from utils.prepare_data import prepare_data
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

NSL_KDD_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/'
NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/test_cs.csv'

BEST_CHECKPOINT = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/model/best_checkpoint/9740'

# Attack Category Model
input = tflearn.input_data(shape=[None, 121], name='input')

hl_1 = tflearn.fully_connected(input, 128, activation='relu', bias=True, name='hl_1')
hl_2 = tflearn.fully_connected(hl_1, 128, activation='relu', bias=True, name='hl_2')
hl_3 = tflearn.fully_connected(hl_2, 128, activation='relu', bias=True, name='hl_3')
hl_4 = tflearn.fully_connected(hl_3, 128, activation='relu', bias=True, name='hl_4')
hl_5 = tflearn.fully_connected(hl_4, 128, activation='relu', bias=True, name='hl_5')
hl_6 = tflearn.fully_connected(hl_5, 128, activation='relu', bias=True, name='hl_6')

ac_output = tflearn.fully_connected(hl_6, 5, activation='softmax', bias=True, name='ac_output')
network = tflearn.layers.estimator.regression(ac_output, loss='categorical_crossentropy')
model = tflearn.models.dnn.DNN(network)

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
print(classification_report(np.argmax(y_test,axis=1), np.argmax(y_test_pred,axis=1)))
print(confusion_matrix(np.argmax(y_test,axis=1), np.argmax(y_test_pred,axis=1)))
np.savetxt(NSL_KDD_DIR + "ac_train_pred.csv", y_train_pred, delimiter=",")
np.savetxt(NSL_KDD_DIR + "ac_val_pred.csv", y_val_pred, delimiter=",")
np.savetxt(NSL_KDD_DIR + "ac_test_pred.csv", y_test_pred, delimiter=",")