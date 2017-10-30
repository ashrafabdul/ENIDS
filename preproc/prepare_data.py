'''
@date:10/10/2017
@author:AshrafAbdul
'''
import pandas as pd

NSL_KDD_MASTER = '/home/aabdul/projects/enids/data/NSL-KDD/master/'
NSL_KDD_MASTER_FILE = '/home/aabdul/projects/enids/data/NSL-KDD/kdd_master_cs.csv'
NSL_KDD_TRAFFIC_TYPE = '/home/aabdul/projects/enids/data/NSL-KDD/traffic_type/'
NSL_KDD_ATTACK_CATEGORY = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/'
NSL_KDD_ATTACK_TYPE = '/home/aabdul/projects/enids/data/NSL-KDD/attack_type/'

MULTI_CATEGORICAL_FEATURES = ['protocol_type','service','flag']
BINARY_CATEGORICAL_FEATURES = ['land','logged_in','root_shell','su_attempted','is_host_login','is_guest_login']
PREDICTIONS = ['attack_type','attack_category','traffic_type']
METADATA = ['difficulty_level','original_split','split']
# Read the master file
df_features = pd.read_csv(NSL_KDD_MASTER_FILE)

df_metadata = df_features[METADATA]
#df_features.head(2)
df_features.drop(METADATA, inplace=True,axis=1)
df_predictions = df_features[PREDICTIONS]
df_features.drop(PREDICTIONS, inplace=True,axis=1)

# Convert categorical features to one hot
cat_features_dfs = []
for feature in MULTI_CATEGORICAL_FEATURES:
    cat_df = pd.get_dummies(df_features[feature], prefix=feature)
    cat_features_dfs.append(cat_df)
    df_features.drop(feature, inplace=True,axis=1)
cat_features_dfs.insert(0, df_features)

cat_predictions_dfs = []
cat_predictions_cols = []
for feature in PREDICTIONS:
    cat_df = pd.get_dummies(df_predictions[feature], prefix=feature)
    cat_df_columns = cat_df.columns
    cat_predictions_dfs.append(cat_df)
    cat_predictions_cols.append(cat_df_columns)
    df_predictions.drop(feature,inplace=True,axis=1)

df_attack_type = pd.concat(cat_features_dfs+[cat_predictions_dfs[0]]+[df_metadata],axis = 1)
df_attack_type.to_csv(NSL_KDD_ATTACK_TYPE+'master_cs.csv',index_label='data_index')
df_attack_type.drop(['original_split','difficulty_level'],inplace=True,axis=1)
df_attack_type_train = df_attack_type.loc[df_attack_type['split'] == "Training"]
df_attack_type_train.drop(['split'],inplace=True,axis=1)
df_attack_type_train.to_csv(NSL_KDD_ATTACK_TYPE+'train_cs.csv',index_label='data_index')
df_attack_type_val = df_attack_type.loc[df_attack_type['split'] == "Validation"]
df_attack_type_val.drop(['split'],inplace=True,axis=1)
df_attack_type_val.to_csv(NSL_KDD_ATTACK_TYPE+'val_cs.csv',index_label='data_index')
df_attack_type_test = df_attack_type.loc[df_attack_type['split'] == "Test"]
df_attack_type_test.drop(['split'],inplace=True,axis=1)
df_attack_type_test.to_csv(NSL_KDD_ATTACK_TYPE+'test_cs.csv',index_label='data_index')

# df_master = pd.concat(cat_features_dfs+cat_predictions_dfs+[df_metadata],axis = 1)
# df_master.to_csv(NSL_KDD_MASTER+'master_cs.csv',index_label='data_index')
# df_master.drop(['original_split','difficulty_level'],inplace=True,axis=1)
# df_master_train = df_master.loc[df_master['split'] == "Training"]
# df_master_train.drop(['split'],inplace=True,axis=1)
# df_master_train.to_csv(NSL_KDD_MASTER+'train_cs.csv',index_label='data_index')
# df_master_val = df_master.loc[df_master['split'] == "Validation"]
# df_master_val.drop(['split'],inplace=True,axis=1)
# df_master_val.to_csv(NSL_KDD_MASTER+'val_cs.csv',index_label='data_index')
# df_master_test = df_master.loc[df_master['split'] == "Test"]
# df_master_test.drop(['split'],inplace=True,axis=1)
# df_master_test.to_csv(NSL_KDD_MASTER+'test_cs.csv',index_label='data_index')

#
df_attack_category = pd.concat(cat_features_dfs+[cat_predictions_dfs[1]]+[df_metadata],axis = 1)
df_attack_category.to_csv(NSL_KDD_ATTACK_CATEGORY+'master_cs.csv',index_label='data_index')
df_attack_category.drop(['original_split','difficulty_level'],inplace=True,axis=1)
df_attack_category_train = df_attack_category.loc[df_attack_category['split'] == "Training"]
df_attack_category_train.drop(['split'],inplace=True,axis=1)
df_attack_category_train.to_csv(NSL_KDD_ATTACK_CATEGORY+'train_cs.csv',index_label='data_index')
df_attack_category_val = df_attack_category.loc[df_attack_category['split'] == "Validation"]
df_attack_category_val.drop(['split'],inplace=True,axis=1)
df_attack_category_val.to_csv(NSL_KDD_ATTACK_CATEGORY+'val_cs.csv',index_label='data_index')
df_attack_category_test = df_attack_category.loc[df_attack_category['split'] == "Test"]
df_attack_category_test.drop(['split'],inplace=True,axis=1)
df_attack_category_test.to_csv(NSL_KDD_ATTACK_CATEGORY+'test_cs.csv',index_label='data_index')
#
df_traffic_type = pd.concat(cat_features_dfs+[cat_predictions_dfs[2]]+[df_metadata],axis = 1)
df_traffic_type.to_csv(NSL_KDD_TRAFFIC_TYPE+'master_cs.csv',index_label='data_index')
df_traffic_type.drop(['original_split','difficulty_level'],inplace=True,axis=1)
df_traffic_type_train = df_traffic_type.loc[df_traffic_type['split'] == "Training"]
df_traffic_type_train.drop(['split'],inplace=True,axis=1)
df_traffic_type_train.to_csv(NSL_KDD_TRAFFIC_TYPE+'train_cs.csv',index_label='data_index')
df_traffic_type_val = df_traffic_type.loc[df_traffic_type['split'] == "Validation"]
df_traffic_type_val.drop(['split'],inplace=True,axis=1)
df_traffic_type_val.to_csv(NSL_KDD_TRAFFIC_TYPE+'val_cs.csv',index_label='data_index')
df_traffic_type_test = df_traffic_type.loc[df_traffic_type['split'] == "Test"]
df_traffic_type_test.drop(['split'],inplace=True,axis=1)
df_traffic_type_test.to_csv(NSL_KDD_TRAFFIC_TYPE+'test_cs.csv',index_label='data_index')