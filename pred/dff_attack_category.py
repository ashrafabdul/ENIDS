'''
@date:14/10/2017
@author:AshrafAbdul
'''
from utils.get_best_model import get_best_model
from utils.prepare_data import prepare_data

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np

NSL_KDD_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/'
NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/test_cs.csv'

ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']

if __name__=="__main__":

    [[x_train, y_train], [x_val, y_val], [x_test, y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)


    model = get_best_model("ac_hl6")

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    for y_pred,y_true,split in [[y_train_pred,y_train,'train'],[y_val_pred,y_val,'val'],[y_test_pred,y_test,'test']]:
            print('===================================================================================')
            print("Split",split)
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
            print(classification_report(y_true,y_pred ,target_names=ATTACK_CATEGORY_COLS))
            print(confusion_matrix(y_true,y_pred))
            print('===================================================================================')
