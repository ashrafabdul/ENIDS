import pandas as pd
import scipy
from scipy.stats import spearmanr
import numpy as np


IG_MTL = '/home/aabdul/projects/enids/data/NSL-KDD/report/ig/mtl.csv'
IG_MLP = '/home/aabdul/projects/enids/data/NSL-KDD/report/ig/mlp.csv'


if __name__=="__main__":

    exp_files = [IG_MTL, IG_MLP]
    model = ['mtl', 'mlp']
    classes = [0,1,2,3,4]
    easy_names = ['dos', 'normal', 'probe', 'r2l', 'u2r']

    df_mtl_f= pd.read_csv(IG_MTL)
    for c in classes:
        df_mtl = df_mtl_f.loc[df_mtl_f['explanation_class'] == c]
        df_mtl.drop(['dindex','explanation_class','cindex'],inplace=True,axis=1)
        df_mtl = pd.DataFrame(df_mtl.columns[np.argsort(-df_mtl.values, axis=1)],index=df_mtl.index)
        df_mtl.to_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/ig/mtl_sorted_'+easy_names[c]+'.csv',index=False)


    df_mlp_f = pd.read_csv(IG_MLP)
    for c in classes:
        df_mlp = df_mlp_f.loc[df_mlp_f['explanation_class'] == c]
        df_mlp.drop(['dindex','explanation_class','cindex'],inplace=True,axis=1)
        df_mlp = pd.DataFrame(df_mlp.columns[np.argsort(-df_mlp.values, axis=1)],index=df_mlp.index)
        df_mlp.to_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/ig/mlp_sorted_' + easy_names[c] + '.csv',index=False)
