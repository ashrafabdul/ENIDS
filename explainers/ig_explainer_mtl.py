from explainers.integrated_gradients import *
from utils.get_best_model import get_best_model_keras
import numpy as np
import pandas as pd
import os
from utils.prepare_data import prepare_data_pandas


NSL_KDD_DIR = '/home/aabdul/projects/enids/data/NSL-KDD/master/'
NSL_KDD_TRAIN = '/home/aabdul/projects/enids/data/NSL-KDD/master/train_cs.csv'
NSL_KDD_VAL = '/home/aabdul/projects/enids/data/NSL-KDD/master/val_cs.csv'
NSL_KDD_TEST = '/home/aabdul/projects/enids/data/NSL-KDD/master/test_cs.csv'
ATTACK_CATEGORY_COLS = ['attack_category_dos','attack_category_normal','attack_category_probe','attack_category_r2l','attack_category_u2r']


ac_predictions_dos_ref = [0,0,105,255,0.86,0,1,0,0,1,0,0,0,2630,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,145,0,0,0,0,0,0,0]
ac_predictions_normal_ref = [1,0,15,255,0.83,0.81,0,0.06,0,16,0,0,0,8181,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
ac_predictions_probe_ref = [0,0,105,255,0.86,0,1,0,0,1,0,0,0,2630,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,145,0,0,0,0,0,0,0]
ac_predictions_r2l_ref = [0,0,105,255,0.86,0,1,0,0,1,0,0,0,2630,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,145,0,0,0,0,0,0,0]
ac_predictions_u2r_ref = [0,0,105,255,0.86,0,1,0,0,1,0,0,0,2630,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,145,0,0,0,0,0,0,0]



model = get_best_model_keras('mtl')
ig = integrated_gradients(model)
ref_vals = [ac_predictions_dos_ref,ac_predictions_normal_ref,ac_predictions_probe_ref,ac_predictions_r2l_ref,ac_predictions_u2r_ref]


x,y = prepare_data_pandas(NSL_KDD_TRAIN,NSL_KDD_VAL,NSL_KDD_TEST,test_only=True)
header = list(x.columns)
f = open('/home/aabdul/projects/enids/data/NSL-KDD/selected_points/'+'mtl_ac_ig.csv','ab')
header = ['index'] + header + ['actual_class','predicted_class'] + [s+"_ig" for s in header] + ['explanation_class']
f.write(str.encode(','.join(header)+'\n'))
x = x.as_matrix()
y = y.as_matrix()
for i in [275,405,1103,1221,6153,27052]:
    actual_class = np.argmax(y[i,1:6].reshape([1,5]))
    predicted_class = np.argmax(model.predict(x[i].reshape([1,121])))
    for j in range(0,5):
        exp = ig.explain(sample=x[i], outc=j, reference=np.array(ref_vals[j]), num_steps=500, verbose=1)
        row = np.hstack(([i],x[i],[actual_class, predicted_class],exp,[j]))
        np.savetxt(f,row.reshape(1, row.shape[0]),delimiter=',')
    print(i)
f.close()



# sample = ac_predictions_normal_ref
# for reference in ref_vals:
#     print(ig.explain(sample=np.array(sample),outc=0,reference=np.array(reference),num_steps=150,verbose=1))
#
#
#


#Num of 0 val features
# basepath = '/home/aabdul/projects/enids/data/NSL-KDD/master/mtl_refs/'
# for f in os.listdir(basepath):
#      if 'min' not in f and 'zero' not in f:
#          df = pd.read_csv(basepath+f)
#          df['zeros'] = (df.loc[:,'count':'wrong_fragment'] == 0 ).astype(int).sum(axis=1)
#          df['sum'] = df.loc[:, 'count':'wrong_fragment'].sum(axis=1)
#          df.to_csv(basepath+f[:-4]+'_zeros_sum.csv',index=False)


# Min of each column
# basepath = '/home/aabdul/projects/enids/data/NSL-KDD/attack_category/ac_hl6_refs/'
# for f in os.listdir(basepath):
#     df = pd.read_csv(basepath+f)
#     cols = df.columns
#     min_values = df.min(axis=0).tolist()
#     print(f,min_values)
#     df = pd.DataFrame(np.array([min_values]))
#     df.columns = cols
#     df.to_csv(basepath+f[:-4]+'_min.csv',index=False)
