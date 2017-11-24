import pandas as pd


IG_MTL = '/home/aabdul/projects/enids/data/NSL-KDD/report/lime/mtl_ac_lime.csv'
IG_MLP = '/home/aabdul/projects/enids/data/NSL-KDD/report/lime/mlp_ac_lime.csv'



#FEATURE_NAMES = ['land','logged_in','root_shell','su_attempted','is_host_login','is_guest_login','protocol_type','service','flag']
#CATEGORICAL_FEATURES = [3,8,10,11,16,17,37,38,39]

def decode_onehot(f):
    df = pd.read_csv(f)
    df = df.fillna(value=0)
    cols = df.columns

    service = [c for c in cols if c.startswith('service')]
    protocol = [c for c in cols if c.startswith('protocol')]
    flag = [c for c in cols if c.startswith('flag')]
    land = [c for c in cols if c.startswith('land')]
    logged_in = [c for c in cols if c.startswith('logged_in')]
    root_shell = [c for c in cols if c.startswith('root')]
    su_attempted = [c for c in cols if c.startswith('su_')]
    is_guest_login = [c for c in cols if c.startswith('is_guest')]
    is_host_login = [c for c in cols if c.startswith('is_host')]

    df['service'] = df[service].sum(axis=1)
    df['protocol_type'] =  df[protocol].sum(axis=1)
    df['flag'] = df[flag].sum(axis=1)
    df['land'] = df[land].sum(axis=1)
    df['logged_in'] = df[logged_in].sum(axis=1)
    df['root_shell'] = df[root_shell].sum(axis=1)
    df['su_attempted'] = df[su_attempted].sum(axis=1)
    df['is_guest_login'] = df[is_guest_login].sum(axis=1)
    df['is_host_login'] = df[is_host_login].sum(axis=1)

    dropcols = service + protocol + flag + land + logged_in + root_shell + su_attempted + ['actual_class'] + is_guest_login + is_host_login
    df.drop(dropcols,axis=1,inplace=True)

    return df

if __name__ == '__main__':

    exp_files = [IG_MTL,IG_MLP]
    model = ['mtl','mlp']

    args = zip(exp_files,model)

    for f,m in args:
        df = decode_onehot(f)
        df.to_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/lime/'+m+'.csv',index=False)
