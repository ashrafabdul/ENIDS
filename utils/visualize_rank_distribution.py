import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-deep')


def plot(x,y,min,max,title,x_label = 'x',y_label = 'y',axis = None):
    data = np.vstack([x, y]).T
    bins = np.linspace(min, max, max-min+1)

    plt.hist(data, bins, alpha=0.7, label=[x_label,y_label])
    #plt.hist(y, bins, alpha=0.7, label=y_label)
    plt.legend(loc='upper right')
    plt.xlabel('order')
    plt.ylabel('frequency')
    plt.axis(axis)
    plt.title(title)
    plt.savefig('../../data/NSL-KDD/spr/new features/img/'+title+"_"+x_label+"_"+y_label+'.png')
    plt.show()

def to_sample(list):
    result = []
    for i in range(0,len(list)):
        n = list[i]
        for j in range(0,n):
            result.append(i+1)
    return result

def compare(df1,df2,name1,name2,axis=[0,41,0,15000]):
    header = list(df1.columns.values)
    for col in header:
        l1 = list(df1[col])
        l2 = list(df2[col])
        bin = len(l1)
        l1 = to_sample(l1)
        l2 = to_sample(l2)
        print (l1)
        print (l2)
        plot(l1,l2,1,bin,col,name1,name2,axis)

if __name__=="__main__":
    ig_mlp = pd.read_csv("../../data/NSL-KDD/spr/new features/IG/ig_mlp_distribution.csv")
    ig_mtl = pd.read_csv("../../data/NSL-KDD/spr/new features/IG/ig_mtl_distribution.csv")
    lime_mlp = pd.read_csv("../../data/NSL-KDD/spr/new features/LIME/lime_mlp_distribution.csv")
    lime_mtl = pd.read_csv("../../data/NSL-KDD/spr/new features/LIME/lime_mtl_distribution.csv")

    compare(ig_mlp,ig_mtl,name1="ig_mlp",name2='ig_mtl')
    compare(lime_mlp,lime_mtl,name1='lime_mlp',name2 = 'lime_mtl',axis = (0,41,0,6000))