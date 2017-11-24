from utils.get_best_model import get_best_model
from utils.prepare_data import prepare_data_pandas

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

NSL_KDD_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/'
NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/master/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/master/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/master/test_cs.csv'

TRAFFIC_TYPE_COLS = ['traffic_type_anomalous','traffic_type_normal']
ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']
ATTACK_TYPE_COLS = ['attack_type_apache2','attack_type_back','attack_type_buffer_overflow','attack_type_ftp_write','attack_type_guess_passwd','attack_type_httptunnel','attack_type_imap','attack_type_ipsweep','attack_type_land','attack_type_loadmodule','attack_type_mailbomb','attack_type_mscan','attack_type_multihop','attack_type_named','attack_type_neptune','attack_type_nmap','attack_type_normal','attack_type_perl','attack_type_phf','attack_type_pod','attack_type_portsweep','attack_type_processtable','attack_type_ps','attack_type_rootkit','attack_type_saint','attack_type_satan','attack_type_sendmail','attack_type_smurf','attack_type_snmpgetattack','attack_type_snmpguess','attack_type_spy','attack_type_sqlattack','attack_type_teardrop','attack_type_udpstorm','attack_type_warezclient','attack_type_warezmaster','attack_type_worm','attack_type_xlock','attack_type_xsnoop','attack_type_xterm']



def split_task_predictions(y_pred, y_true):
    y_pred_tt = y_pred[:,0:2]
    y_true_tt = y_true[:,0:2]

    y_pred_ac = y_pred[:,2:7]
    y_true_ac = y_true[:,2:7]

    y_pred_at = y_pred[:,7:40]
    y_true_at = y_true[:,7:40]

    return([[y_pred_tt,y_true_tt,'traffic_type',TRAFFIC_TYPE_COLS],[y_pred_ac,y_true_ac,'attack_category',ATTACK_CATEGORY_COLS]])

if __name__=="__main__":

    x,y = prepare_data_pandas(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)
    x_cols = x.columns
    y_cols = y.columns
    model = get_best_model("mtl")


    y_pred_cols = [s+'_pred' for s in ATTACK_CATEGORY_COLS]

    batch_x = np.array_split(x.as_matrix(),3)
    y_pred = []
    for batch in batch_x:
        pred = model.predict(batch)
        y_pred.append(pred[:,2:7])

    df_ypred = pd.DataFrame(np.vstack(y_pred))
    df_ypred.columns = y_pred_cols
    print(len(df_ypred))
    print(len(x))
    print(len(y))
    df = pd.concat([x,y,df_ypred],ignore_index=True,axis=1)

    df.columns = list(x_cols) + list(y_cols) + y_pred_cols
    df.to_csv(NSL_KDD_DIR+'_mtl_ac_predictions_full.csv',index=False)


    # [[x_train, y_train], [x_val, y_val], [x_test, y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)
    #
    # model = get_best_model("mtl_tt_ac")
    #
    # y_train_pred = model.predict(x_train)
    # y_val_pred = model.predict(x_val)
    # y_test_pred = model.predict(x_test)
    #
    # for y_pred,y_true,split in [[y_train_pred,y_train,'train'],[y_val_pred,y_val,'val'],[y_test_pred,y_test,'test']]:
    #     task_splits = split_task_predictions(y_pred,y_true)
    #     for y_pred_task,y_true_task,task,labels in task_splits:
    #         print('===================================================================================')
    #         print("Split",split,"Task",task)
    #         y_true_task = np.argmax(y_true_task, axis=1)
    #         y_pred_task = np.argmax(y_pred_task, axis=1)
    #         print(classification_report(y_true_task,y_pred_task ,target_names=labels))
    #         y_true_task_labels = []
    #         y_pred_task_labels = []
    #         for v in y_true_task:
    #             y_true_task_labels.append(labels[v])
    #         for v in y_pred_task:
    #             y_pred_task_labels.append(labels[v])
    #         print(confusion_matrix(y_true_task_labels,y_pred_task_labels,labels=labels))
    #         print('===================================================================================')
