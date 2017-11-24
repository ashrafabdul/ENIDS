import pandas as pd
import scipy
from scipy.stats import spearmanr
import numpy as np


BASEPATH = '/home/aabdul/projects/enids/data/NSL-KDD/report/ig/'

if __name__=="__main__":

    models = ['mtl', 'mlp']
    acs = ['dos', 'normal', 'probe', 'r2l', 'u2r']
    for m in models:
        for ac in acs:
            sff = m+'_sorted_'+ac+'.csv'
            df = pd.read_csv(BASEPATH+sff)
            features = list(set(df.iloc[0]))
            ranks = []
            for feature in features:
                df[feature] = (df == feature).idxmax(axis=1)

            cols = df.columns
            cols = [c for c in cols if c in features]
            df = df[cols].astype(int,copy=True)
            # Add 1 since rank starts from 1
            df[cols] += 1

            feature_frequencies = []
            for feature in cols:
                feature_frequencies.append(df[feature].value_counts())
            df_ff = pd.concat(feature_frequencies,axis=1)
            df_ff.fillna(value=0,axis=1,inplace=True)


            df_ff.to_csv(BASEPATH+m+'_'+ac+'_feature_rank_distribution.csv',index=False)






