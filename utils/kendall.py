import pandas as pd
import scipy
from scipy.stats import spearmanr,kendalltau
import numpy as np

def rank(path,drop = ['dindex','explanation_class', 'cindex']):
    df = pd.read_csv(path)
    df = df.drop(drop, inplace=False, axis=1)

    return df

def kendall(df1,df2):

    result = []
    for t in range(0,5):
        result.append(pd.DataFrame(columns=['tau','p_value']))
    print (result)

    for i in range(df1.shape[0]):
        if i%1000 ==0:
            print (i)
        v1 = df1.loc[i]
        v2 = df2.loc[i]
        v1 = v1.rank(method = 'first')#method = 'average')
        v2 = v2.rank(method = 'first')#method = 'average')
        tau,p_value = kendalltau(v1,v2)
        result[i%5].loc[i//5] = [tau,p_value]
        #print ([tau,p_value])
    return result

if __name__=="__main__":

    df_mtl_ig = rank(path = '../../data/NSL-KDD/spr/new features/IG/mtl.csv')
    df_mtl_lime = rank(path = '../../data/NSL-KDD/spr/new features/LIME/mtl.csv',\
                       drop=['dindex','explanation_class'])
    df_mlp_ig = rank(path = '../../data/NSL-KDD/spr/new features/IG/mlp.csv')
    df_mlp_lime = rank(path = '../../data/NSL-KDD/spr/new features/LIME/mlp.csv',\
                       drop=['dindex','explanation_class'])


    print ("data loaded")
    result_mtl = kendall(df_mtl_ig,df_mtl_lime)
    print("kendall finished for mtl in class "+str(type))
    #print(result_mtl)
    result_mlp = kendall(df_mlp_ig,df_mlp_lime)
    print("kendall finished for mlp in class "+str(type))
    #print(result_mlp)
    #print ("result:",result)

    for type in range(0,5):
        mean_mtl = np.mean(result_mtl[type],axis=0)
        print ("mean_mtl "+str(type)+" :",mean_mtl)
        result_mtl[type].loc[result_mtl[type].shape[0]] = [mean_mtl[0],mean_mtl[1]]
        result_mtl[type].to_csv("../../data/NSL-KDD/spr/new features/mtl_kendall_"+str(type)+".csv")

        mean_mlp = np.mean(result_mlp[type],axis=0)
        print ("mean_mlp "+str(type)+" :",mean_mlp)
        result_mlp[type].loc[result_mlp[type].shape[0]] = [mean_mlp[0],mean_mlp[1]]
        result_mlp[type].to_csv("../../data/NSL-KDD/spr/new features/mlp_kendall_"+str(type)+".csv")

'''
    s = spearmanr(df_mtl,df_mlp)
    s_mean = np.mean(s, axis=1)
    fs = open('../../data/NSL-KDD/spr/spr.txt','wb')
    fsm = open('../../data/NSL-KDD/spr/spr_1.txt', 'wb')
    np.savetxt(fs,s)
    np.savetxt(fsm, s_mean)
'''