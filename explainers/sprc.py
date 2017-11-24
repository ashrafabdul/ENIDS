import pandas as pd
import scipy
from scipy.stats import spearmanr,kendalltau
import numpy as np

def rank(df,drop = ['dindex', 'explanation_class', 'cindex']):
    df = df.drop(drop, inplace=False, axis=1)
    df = pd.DataFrame(df.columns[np.argsort(-df.values, axis=1)],index=df.index)
    df = df.as_matrix()
    return df

if __name__=="__main__":

    df_mtl_ig = rank(pd.read_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/ig/mtl.csv'))
    df_mtl_lime = rank(pd.read_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/lime/mtl.csv'),drop = ['dindex', 'explanation_class'])
    df_mlp_ig = rank(pd.read_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/ig/mlp.csv'))
    df_mlp_lime = rank(pd.read_csv('/home/aabdul/projects/enids/data/NSL-KDD/report/lime/mlp.csv'),drop = ['dindex', 'explanation_class'])

    print ("data loaded")


    s_mtl = spearmanr(df_mtl_ig,df_mtl_lime)
    print ("spearman finished for mtl")
    s_mtl_mean = np.mean(s_mtl, axis=1)
    fs_mtl = open('/home/aabdul/projects/enids/data/NSL-KDD/report/mtl_spearman.txt','wb')
    fsm_mtl = open('/home/aabdul/projects/enids/data/NSL-KDD/report/mtl_spearman_mean.txt', 'wb')
    np.savetxt(fs_mtl,s_mtl)
    np.savetxt(fsm_mtl, s_mtl_mean)


    s_mlp = spearmanr(df_mlp_ig,df_mlp_lime)
    print("spearman finished for mlp")
    s_mlp_mean = np.mean(s_mlp, axis=1)
    fs_mlp = open('/home/aabdul/projects/enids/data/NSL-KDD/report/mlp_spearman.txt','wb')
    fsm_mlp = open('/home/aabdul/projects/enids/data/NSL-KDD/report/mlp_spearman_mean.txt', 'wb')
    np.savetxt(fs_mlp,s_mlp)
    np.savetxt(fsm_mlp, s_mlp_mean)