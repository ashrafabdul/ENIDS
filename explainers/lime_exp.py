'''
@date:14/10/2017
@author:AshrafAbdul
'''

import lime.lime_tabular
from scipy.special import entr
from sklearn.externals import joblib
import numpy as np
import re
import random
import tflearn
import tensorflow as tf
from utils.prepare_data import prepare_data
import numpy as np
import pandas as pd
import sklearn
from utils.get_best_model import get_best_model
import copy


NSL_KDD_MASTER = '/home/aabdul/projects/enids/data/NSL-KDD/kdd_master_cs_lime.csv'
NSL_KDD_MASTER_TRUE_LABELS = '/home/aabdul/projects/enids/data/NSL-KDD/master/master_cs.csv'

TRAFFIC_TYPE_COLS = ['traffic_type_anomalous','traffic_type_normal']
ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']
ATTACK_TYPE_COLS = ['attack_type_apache2','attack_type_back','attack_type_buffer_overflow','attack_type_ftp_write','attack_type_guess_passwd','attack_type_httptunnel','attack_type_imap','attack_type_ipsweep','attack_type_land','attack_type_loadmodule','attack_type_mailbomb','attack_type_mscan','attack_type_multihop','attack_type_named','attack_type_neptune','attack_type_nmap','attack_type_normal','attack_type_perl','attack_type_phf','attack_type_pod','attack_type_portsweep','attack_type_processtable','attack_type_ps','attack_type_rootkit','attack_type_saint','attack_type_satan','attack_type_sendmail','attack_type_smurf','attack_type_snmpgetattack','attack_type_snmpguess','attack_type_spy','attack_type_sqlattack','attack_type_teardrop','attack_type_udpstorm','attack_type_warezclient','attack_type_warezmaster','attack_type_worm','attack_type_xlock','attack_type_xsnoop','attack_type_xterm']
METADATA_COLS = ['difficulty_level', 'original_split', 'split']
LIME_INPUT_DROP_COLS = ['attack_type', 'difficulty_level', 'attack_category', 'traffic_type', 'original_split', 'split']

FEATURE_NAMES = ['duration','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','protocol_type','service','flag']
CATEGORICAL_FEATURES = [3,8,10,11,16,17,37,38,39]
MULTI_CATEGORICAL_FEATURES = ['protocol_type','service','flag']
DNN_FEATURE_ORDER = ['count', 'diff_srv_rate', 'dst_bytes', 'dst_host_count', 'dst_host_diff_srv_rate', 'dst_host_rerror_rate', 'dst_host_same_src_port_rate', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 'dst_host_srv_count', 'dst_host_srv_diff_host_rate', 'dst_host_srv_rerror_rate', 'dst_host_srv_serror_rate', 'duration', 'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH', 'hot', 'is_guest_login', 'is_host_login', 'land', 'logged_in', 'num_access_files', 'num_compromised', 'num_failed_logins', 'num_file_creations', 'num_root', 'num_shells', 'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp', 'rerror_rate', 'root_shell', 'same_srv_rate', 'serror_rate', 'service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 'service_auth', 'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard', 'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs', 'service_exec', 'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest', 'service_hostnames', 'service_http', 'service_http_2784', 'service_http_443', 'service_http_8001', 'service_imap4', 'service_iso_tsap', 'service_klogin', 'service_kshell', 'service_ldap', 'service_link', 'service_login', 'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns', 'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp', 'service_ntp_u', 'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private', 'service_red_i', 'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net', 'service_ssh', 'service_sunrpc', 'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i', 'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp', 'service_uucp_path', 'service_vmnet', 'service_whois', 'src_bytes', 'srv_count', 'srv_diff_host_rate', 'srv_rerror_rate', 'srv_serror_rate', 'su_attempted', 'urgent', 'wrong_fragment']

SERVICE_MAP = {0: 'IRC', 1: 'X11', 2: 'Z39_50', 3: 'aol', 4: 'auth', 5: 'bgp', 6: 'courier', 7: 'csnet_ns', 8: 'ctf', 9: 'daytime', 10: 'discard', 11: 'domain', 12: 'domain_u', 13: 'echo', 14: 'eco_i', 15: 'ecr_i', 16: 'efs', 17: 'exec', 18: 'finger', 19: 'ftp', 20: 'ftp_data', 21: 'gopher', 22: 'harvest', 23: 'hostnames', 24: 'http', 25: 'http_2784', 26: 'http_443', 27: 'http_8001', 28: 'imap4', 29: 'iso_tsap', 30: 'klogin', 31: 'kshell', 32: 'ldap', 33: 'link', 34: 'login', 35: 'mtp', 36: 'name', 37: 'netbios_dgm', 38: 'netbios_ns', 39: 'netbios_ssn', 40: 'netstat', 41: 'nnsp', 42: 'nntp', 43:'ntp_u', 44:'other', 45:'pm_dump', 46:'pop_2', 47:'pop_3', 48:'printer', 49:'private', 50:'red_i', 51:'remote_job', 52:'rje', 53:'shell', 54:'smtp', 55:'sql_net', 56:'ssh', 57:'sunrpc', 58:'supdup', 59:'systat', 60:'telnet', 61:'tftp_u', 62:'tim_i', 63:'time', 64:'urh_i', 65:'urp_i', 66:'uucp', 67:'uucp_path', 68:'vmnet', 69:'whois'}
PROTOCOL_MAP = {0: 'icmp', 1: 'tcp', 2: 'udp'}
FLAG_MAP = {0: 'OTH', 1: 'REJ', 2: 'RSTO', 3: 'RSTOS0', 4: 'RSTR', 5: 'S0', 6: 'S1', 7: 'S2', 8: 'S3', 9: 'SF', 10: 'SH'}


CATEGORICAL_NAMES = {}


def lime_init_data():

    global CATEGORICAL_NAMES
    global CATEGORICAL_FEATURES

    data = pd.read_csv(NSL_KDD_MASTER_TRUE_LABELS)
    tt_labels = data[TRAFFIC_TYPE_COLS].values
    ac_labels = data[ATTACK_CATEGORY_COLS].values
    at_labels = data[ATTACK_TYPE_COLS].values

    data = pd.read_csv(NSL_KDD_MASTER)
    index = list(data.index)
    data.drop(LIME_INPUT_DROP_COLS, inplace=True, axis=1)

    data = data.values
    for feature in CATEGORICAL_FEATURES:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        print(set(data[:, feature]))
        CATEGORICAL_NAMES[feature] = le.classes_

    return (data,index, tt_labels, ac_labels, at_labels)

def predict_fn(x):

    # Map data to names
    df = pd.DataFrame(x,columns=FEATURE_NAMES)
    num_points = len(df.index)


    # Reverse label encoding
    df['service'] = df['service'].map(SERVICE_MAP)
    df['flag'] = df['flag'].map(FLAG_MAP)
    df['protocol_type'] = df['protocol_type'].map(PROTOCOL_MAP)

    # One hot encode the labels
    cat_features_dfs = []
    for feature in MULTI_CATEGORICAL_FEATURES:
        cat_df = pd.get_dummies(df[feature], prefix=feature)
        cat_features_dfs.append(cat_df)
        df.drop(feature, inplace=True, axis=1)
    cat_features_dfs.insert(0, df)

    #Merge the one hot encoded features into df
    df = pd.concat(cat_features_dfs,axis = 1)

    # Find missing columns due to onehot encoding on sample
    missing_columns = list(set(DNN_FEATURE_ORDER) - set(df.columns))

    #initialize the missing columns to 0's'
    for col in missing_columns:
        df[col] = [0] * num_points

    # reorder the features as expected by the nn
    df = df[DNN_FEATURE_ORDER]

    return MODEL.predict(df.values)



def examine_points(x):
    # Map data to names
    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    num_points = len(df.index)

    # Reverse label encoding
    df['service'] = df['service'].map(SERVICE_MAP)
    df['flag'] = df['flag'].map(FLAG_MAP)
    df['protocol_type'] = df['protocol_type'].map(PROTOCOL_MAP)

    # One hot encode the labels
    cat_features_dfs = []
    for feature in MULTI_CATEGORICAL_FEATURES:
        cat_df = pd.get_dummies(df[feature], prefix=feature)
        cat_features_dfs.append(cat_df)
        df.drop(feature, inplace=True, axis=1)
    cat_features_dfs.insert(0, df)

    # Merge the one hot encoded features into df
    df = pd.concat(cat_features_dfs, axis=1)

    # Find missing columns due to onehot encoding on sample
    missing_columns = list(set(DNN_FEATURE_ORDER) - set(df.columns))

    # initialize the missing columns to 0's'
    for col in missing_columns:
        df[col] = [0] * num_points

    # reorder the features as expected by the nn
    df = df[DNN_FEATURE_ORDER]

    # For now return points with
    # a) high entropy -> most confused
    # b) zero entropy -> most confident
    # extend later to within class points

    # Calculate entropy
    p = MODEL.predict(df.values)
    e = entr(p)
    e = np.sum(e,axis=1)

    #Calculate standard deviation of entropy
    std = np.std(e)
    high_entropy_index = np.where(e > 1*std )

    zero_entropy_index = np.where(e==0)

    return zero_entropy_index,high_entropy_index, p




if __name__=="__main__":

    MODEL = None

    (data,data_index,tt_labels, ac_labels, at_labels) = lime_init_data()
    print(ac_labels)

    models = ['tt_hl6','ac_hl6','at_hl6']
    classes = [TRAFFIC_TYPE_COLS,ATTACK_CATEGORY_COLS,ATTACK_TYPE_COLS]
    num_exp_labels = [1,3,3]
    true_labels = (tt_labels, ac_labels, at_labels)
    exp_dirs = ['/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/explanations/','/home/aabdul/projects/enids/data/NSL-KDD/attack_category/explanations/','/home/aabdul/projects/enids/data/NSL-KDD/attack_type/explanations/']

    args = zip(models, classes, num_exp_labels,true_labels,exp_dirs)
    for model_name, class_names, top_labels, true_label, dp in args:

        del(MODEL)
        MODEL = get_best_model(model_name)
        explainer = lime.lime_tabular.LimeTabularExplainer(data, feature_names=FEATURE_NAMES, class_names=class_names, categorical_features=CATEGORICAL_FEATURES, categorical_names=CATEGORICAL_NAMES, discretize_continuous=True)
        sys.exit()
        zero_entropy_index, high_entropy_index, pred_label = examine_points(copy.deepcopy(data))

        #print("Number of confident points:", zero_entropy_index[0])
        #print("Number of confused points:", high_entropy_index[0])

        #print("Number of confident points:", len(zero_entropy_index[0]))
        #print("Number of confused points:", len(high_entropy_index[0]))

        true_label = np.argmax(true_label,axis=1)
        pred_label = np.argmax(pred_label,axis=1)

        #np.savetxt('/home/aabdul/tmp/'+model_name +'_true_label.csv', true_label, delimiter=",")
        #np.savetxt('/home/aabdul/tmp/'+model_name +'_pred_label.csv', pred_label, delimiter=",")

        #len_z = len()
        #len_h = len())

        # num_sample = random.sample(range(1,len(zero_entropy_index[0])),10)
        # for i in num_sample:
        #     exp = explainer.explain_instance(data[i], predict_fn, num_features=8, top_labels=top_labels)
        #     if true_label[i] != pred_label[i]:
        #          exp.save_to_file(dp+'confident/misclassified/' + str(i) + '.html')
        #     else:
        #          exp.save_to_file(dp+'confident/classified/' + str(i) + '.html')

        # num_sample = random.sample(range(1, len(high_entropy_index[0])), 10)
        # for i in  num_sample:
        #      exp = explainer.explain_instance(data[i], predict_fn, num_features=8, top_labels=top_labels)
        #      if true_label[i] != pred_label[i]:
        #          exp.save_to_file(dp+'confused/misclassified/' + str(i) + '.html')
        #      else:
        #          exp.save_to_file(dp+'confused/classified/' + str(i) + '.html')