from matplotlib import pyplot
import numpy as np


LIME_BASEPATH = '/home/aabdul/projects/enids/data/NSL-KDD/report/lime/'
IG_BASEPATH = '/home/aabdul/projects/enids/data/NSL-KDD/report/ig/'

if __name__=="__main__":

    acs = ['dos', 'normal', 'probe', 'r2l', 'u2r']
    bins = np.linspace(1,40,40)
    print(bins)