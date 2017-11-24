import pandas as pd


IG_MTL = '/home/aabdul/projects/enids/data/NSL-KDD/spr/mtl_ig_explanations_test_spr.csv'
IG_MLP = '/home/aabdul/projects/enids/data/NSL-KDD/spr/mlp_ig_explanations_test_spr.csv'



#FEATURE_NAMES = ['land','logged_in','root_shell','su_attempted','is_host_login','is_guest_login','protocol_type','service','flag']
#CATEGORICAL_FEATURES = [3,8,10,11,16,17,37,38,39]

def decode_onehot(f):
    df = pd.read_csv(f)
    cols = df.columns

    service = [c for c in cols if c.startswith('service_') and not c.endswith('_ig')]
    service_ig = [c for c in cols if c.startswith('service_') and c.endswith('_ig')]
    ohr = zip(service,service_ig)
    for s,sig in ohr:
        df[sig] = df[s]*df[sig]

    protocol = [c for c in cols if c.startswith('protocol_') and not c.endswith('_ig')]
    protocol_ig = [c for c in cols if c.startswith('protocol_') and c.endswith('_ig')]
    ohr = zip(protocol, protocol_ig)
    for p, pig in ohr:
        df[pig] = df[p] * df[pig]

    flag = [c for c in cols if c.startswith('flag_') and not c.endswith('_ig')]
    flag_ig = [c for c in cols if c.startswith('flag_') and c.endswith('_ig')]
    ohr = zip(flag, flag_ig)
    for f, fig in ohr:
        df[fig] = df[f] * df[fig]

    df['service_ig'] = df[service_ig].sum(axis=1)
    df['protocol_type_ig'] =  df[protocol_ig].sum(axis=1)
    df['flag_ig'] = df[flag_ig].sum(axis=1)

    dropcols = service_ig + protocol_ig + flag_ig + [c for c in cols if not c.endswith('_ig')]
    for c in ['explanation_class','dindex','cindex']:
        dropcols.remove(c)
    df.drop(dropcols,axis=1,inplace=True)
    cols = df.columns
    cols = [c[:-3] if c not in ['explanation_class','dindex','cindex'] else c for c in cols ]
    df.columns = cols
    return df

if __name__ == '__main__':

    exp_files = [IG_MTL,IG_MLP]
    model = ['mtl','mlp']

    args = zip(exp_files,model)

    for f,m in args:
        df = decode_onehot(f)
        df.to_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/ig/'+m+'.csv',index=False)
