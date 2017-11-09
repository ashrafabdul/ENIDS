'''
@date:14/10/2017
@author:AshrafAbdul
'''
import tflearn
import tensorflow as tf
from utils.prepare_data import prepare_data
import numpy as np

NSL_KDD_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/'
NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/test_cs.csv'



if __name__=="__main__":

    [[x_train, y_train], [x_val, y_val], [x_test, y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)

    model = get_best_model("mtl")

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    for y_pred,y_true,split in [[y_train_pred,y_train,'train'],[y_val_pred,y_val,'val'],[y_test_pred,y_test,'test']]:
        task_splits = split_task_predictions(y_pred,y_true)
        for y_pred_task,y_true_task,task,labels in task_splits:
            print('===================================================================================')
            print("Split",split,"Task",task)
            y_true_task = np.argmax(y_true_task, axis=1)
            y_pred_task = np.argmax(y_pred_task, axis=1)
            print(classification_report(y_true_task,y_pred_task ,target_names=labels))
            y_true_task_labels = []
            y_pred_task_labels = []
            for v in y_true_task:
                y_true_task_labels.append(labels[v])
            for v in y_pred_task:
                y_pred_task_labels.append(labels[v])
            print(confusion_matrix(y_true_task_labels,y_pred_task_labels,labels=labels))
            print('===================================================================================')