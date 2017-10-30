'''
@date:13/10/2017
@author:AshrafAbdul
'''

import tensorflow as tf

s = tf.Session()
input = [[0,1,2,3,4],[5,6,7,8,9]]
input = tf.constant(input,shape=[2,5])
begin = [0,2]
size = [-1,2]

print(input.eval(session=s))
print(tf.slice(input,begin,size).eval(session=s))