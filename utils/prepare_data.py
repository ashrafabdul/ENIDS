'''
@date:14/10/2017
@author:AshrafAbdul
'''

import pandas as pd
import numpy as np

INDEX_COL = ['data_index']
TRAFFIC_TYPE_COLS = ['traffic_type_anomalous','traffic_type_normal']
ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']
ATTACK_TYPE_COLS = ['attack_type_apache2','attack_type_back','attack_type_buffer_overflow','attack_type_ftp_write','attack_type_guess_passwd','attack_type_httptunnel','attack_type_imap','attack_type_ipsweep','attack_type_land','attack_type_loadmodule','attack_type_mailbomb','attack_type_mscan','attack_type_multihop','attack_type_named','attack_type_neptune','attack_type_nmap','attack_type_normal','attack_type_perl','attack_type_phf','attack_type_pod','attack_type_portsweep','attack_type_processtable','attack_type_ps','attack_type_rootkit','attack_type_saint','attack_type_satan','attack_type_sendmail','attack_type_smurf','attack_type_snmpgetattack','attack_type_snmpguess','attack_type_spy','attack_type_sqlattack','attack_type_teardrop','attack_type_udpstorm','attack_type_warezclient','attack_type_warezmaster','attack_type_worm','attack_type_xlock','attack_type_xsnoop','attack_type_xterm']

def prepare_data(train_file,val_file,test_file,standardize_training_data=False,scale=True):

    PRED_COLS = TRAFFIC_TYPE_COLS + ATTACK_CATEGORY_COLS + ATTACK_TYPE_COLS

    if "traffic_type" in train_file:
        PRED_COLS = TRAFFIC_TYPE_COLS
    elif "attack_category" in train_file:
        PRED_COLS = ATTACK_CATEGORY_COLS
    elif "attack_type" in train_file:
        PRED_COLS = ATTACK_TYPE_COLS

    df_train = pd.read_csv(train_file, dtype=np.float32)
    x_train = df_train[df_train.columns.difference(INDEX_COL + PRED_COLS)]
    y_train = df_train[PRED_COLS]
    #print(y_train.head(2))
    if(standardize_training_data):
        if(scale):
            for col in x_train.columns:
                min = x_train[col].min()
                max = x_train[col].max()
                #print(col,max,min)
                if max - min == 0:
                    if not max == 0:
                        x_train[col] = x_train[col]/max
                else:
                    x_train[col] = (x_train[col] - min) / max - min
        else:
            for col in x_train.columns:
                std = x_train[col].std(ddof=0)
                mean = x_train[col].mean()
                if not std == 0:
                    x_train[col] = (x_train[col] - mean)/ std
                else:
                    print(col,std,mean)
    #print(x_train.loc[:, x_train.isnull().any()])
    x_train = x_train.values
    y_train = y_train.values

    df_val = pd.read_csv(val_file)
    x_val = df_val[df_val.columns.difference(INDEX_COL + PRED_COLS)]
    y_val = df_val[PRED_COLS]
    x_val = x_val.values
    y_val = y_val.values

    df_test = pd.read_csv(test_file)
    x_test = df_test[df_test.columns.difference(INDEX_COL + PRED_COLS)]
    y_test = df_test[PRED_COLS]
    x_test = x_test.values
    y_test = y_test.values

    return [[x_train,y_train],[x_val,y_val],[x_test,y_test]]


def prepare_data_pandas(train_file,val_file,test_file,standardize_training_data=False,scale=True,test_only=True):

    PRED_COLS = TRAFFIC_TYPE_COLS + ATTACK_CATEGORY_COLS + ATTACK_TYPE_COLS

    if "traffic_type" in train_file:
        PRED_COLS = TRAFFIC_TYPE_COLS
    elif "attack_category" in train_file:
        PRED_COLS = ATTACK_CATEGORY_COLS
    elif "attack_type" in train_file:
        PRED_COLS = ATTACK_TYPE_COLS

    df_train = pd.read_csv(train_file, dtype=np.float32)
    x_train = df_train[df_train.columns.difference(INDEX_COL + PRED_COLS)]
    y_train = df_train[PRED_COLS]
    #print(y_train.head(2))
    if(standardize_training_data):
        if(scale):
            for col in x_train.columns:
                min = x_train[col].min()
                max = x_train[col].max()
                #print(col,max,min)
                if max - min == 0:
                    if not max == 0:
                        x_train[col] = x_train[col]/max
                else:
                    x_train[col] = (x_train[col] - min) / max - min
        else:
            for col in x_train.columns:
                std = x_train[col].std(ddof=0)
                mean = x_train[col].mean()
                if not std == 0:
                    x_train[col] = (x_train[col] - mean)/ std
                else:
                    print(col,std,mean)
    #print(x_train.loc[:, x_train.isnull().any()])


    df_val = pd.read_csv(val_file)
    x_val = df_val[df_val.columns.difference(INDEX_COL + PRED_COLS)]
    y_val = df_val[PRED_COLS]


    df_test = pd.read_csv(test_file)
    x_test = df_test[df_test.columns.difference(INDEX_COL + PRED_COLS)]
    y_test = df_test[PRED_COLS]

    if not test_only:
        x = pd.concat([x_train,x_val,x_test])
        y = pd.concat([y_train, y_val, y_test])
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
    else:
        x = x_test
        y = y_test
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

    return [x, y]


if __name__=="__main__":
    NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/train_cs.csv'
    NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/val_cs.csv'
    NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/test_cs.csv'
    [x,y] = prepare_data_pandas(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST)
    x.to_csv('/home/aabdul/tmp/prepoc_feature_order_tt_x.csv',index=False)
    y.to_csv('/home/aabdul/tmp/prepoc_feature_order_tt_y.csv', index=False)