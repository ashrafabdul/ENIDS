'''
@date:14/10/2017
@author:AshrafAbdul
'''

from utils.prepare_data import prepare_data_pandas
from scipy.special import entr
import numpy as np
import pandas as pd
from utils.get_best_model import get_best_model_keras

NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/master/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/master/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/master/test_cs.csv'

def points_by_entropy(model,x):

    # Calculate entropy
    p = model.predict(x)
    e = entr(p)
    e = np.sum(e,axis=1)

    #Calculate standard deviation of entropy
    std = np.std(e)
    high_entropy_index = np.where(e > 2*std )

    zero_entropy_index = np.where(e==0)

    return [zero_entropy_index,high_entropy_index]

def points_by_classification(y,y_pred):
    c = np.equal(y, y_pred).all(axis=1)
    print(c)
    classified_index = np.where(c==True)
    misclassified_index = np.where(c==False)

    return[set(classified_index[0]),set(misclassified_index[0])]

if __name__=="__main__":

    x, y = prepare_data_pandas(NSL_KDD_TRAIN, NSL_KDD_VAL, NSL_KDD_TEST, test_only=True)
    x = x.as_matrix()
    y = y.as_matrix()
    y_ac = y[:,2:5]
    y = np.argmax(y_ac,axis=1)


    # MLP AC
    model = get_best_model_keras('ac_hl6')
    confident, confused = points_by_entropy(model,x)
    confident, confused = set(confident[0]), set(confused[0])

    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred,axis=1)

    classified, missclassified = points_by_classification(y.reshape(-1,1),y_pred.reshape(-1,1))

    confident_and_classified = confident.intersection(classified)
    confident_and_misclassified = confident.intersection(missclassified)
    confused_and_classified = confused.intersection(classified)
    confused_and_misclassified = confused.intersection(missclassified)

    print('mlp confident_and_classified:',confident_and_classified.pop())
    print('mlp confident_and_misclassified:',confident_and_misclassified.pop())
    print('mlp confused_and_classified:',confused_and_classified.pop())
    print('mlp confused_and_misclassified:',confused_and_misclassified.pop())

    # MTL AC
    model = get_best_model_keras('mtl_ac')
    confident, confused = points_by_entropy(model, x)
    confident, confused = set(confident[0]), set(confused[0])

    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)

    classified, missclassified = points_by_classification(y.reshape(-1, 1), y_pred.reshape(-1, 1))

    confident_and_classified = confident.intersection(classified)
    confident_and_misclassified = confident.intersection(missclassified)
    confused_and_classified = confused.intersection(classified)
    confused_and_misclassified = confused.intersection(missclassified)

    print('mtl confident_and_classified:', confident_and_classified.pop())
    print('mtl confident_and_misclassified:', confident_and_misclassified.pop())
    print('mtl confused_and_classified:', confused_and_classified.pop())
    print('mtl confused_and_misclassified:', confused_and_misclassified.pop())