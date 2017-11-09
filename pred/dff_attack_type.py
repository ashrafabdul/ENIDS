from utils.get_best_model import get_best_model
from utils.prepare_data import prepare_data

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np

NSL_KDD_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/attack_type/'
NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/attack_type/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/attack_type/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/attack_type/test_cs.csv'

ATTACK_TYPE_COLS = ['attack_type_apache2','attack_type_back','attack_type_buffer_overflow','attack_type_ftp_write','attack_type_guess_passwd','attack_type_httptunnel','attack_type_imap','attack_type_ipsweep','attack_type_land','attack_type_loadmodule','attack_type_mailbomb','attack_type_mscan','attack_type_multihop','attack_type_named','attack_type_neptune','attack_type_nmap','attack_type_normal','attack_type_perl','attack_type_phf','attack_type_pod','attack_type_portsweep','attack_type_processtable','attack_type_ps','attack_type_rootkit','attack_type_saint','attack_type_satan','attack_type_sendmail','attack_type_smurf','attack_type_snmpgetattack','attack_type_snmpguess','attack_type_spy','attack_type_sqlattack','attack_type_teardrop','attack_type_udpstorm','attack_type_warezclient','attack_type_warezmaster','attack_type_worm','attack_type_xlock','attack_type_xsnoop','attack_type_xterm']


if __name__=="__main__":

    [[x_train, y_train], [x_val, y_val], [x_test, y_test]] = prepare_data(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)

    model = get_best_model("at_hl6")

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    for y_pred,y_true,split in [[y_train_pred,y_train,'train'],[y_val_pred,y_val,'val'],[y_test_pred,y_test,'test']]:
            print('===================================================================================')
            print("Split",split)
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
            print(classification_report(y_true,y_pred ,target_names=ATTACK_TYPE_COLS))
            print(confusion_matrix(y_true,y_pred))
            print('===================================================================================')
