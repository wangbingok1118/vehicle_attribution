"""
copy  B stage trained model to D stage pretrain model
conv1 conv2 conv3   cov4 conv 5 : copy B stage trained model
"""
import numpy as np
import sys, os

sys.path.insert(0, './caffe/python')
import caffe


C_stage_proto = './models/C_brand_train.prototxt'
C_stage_model = 'models/C-stage-brand_iter_50000.caffemodel'

D_stage_proto = './models/D_attribution_train.prototxt'
D_stage_model = 'models/D-stage-attribution_iter_64000.caffemodel'


final_proto = './models/final_deploy.prototxt'
final_model = './models/final.caffemodel'

def merge(net_ori,ori_key, net_final, final_key):
    weights = net_ori.params[ori_key]
    for i , w in enumerate(weights):
        net_final.params[final_key][i].data[...] = w.data

    pass

if __name__ == '__main__':
    net_C_stage = caffe.Net(C_stage_proto, C_stage_model, caffe.TRAIN)
    net_D_stage = caffe.Net(D_stage_proto, D_stage_model, caffe.TRAIN)
    net_final = caffe.Net(final_proto, caffe.TEST)
    for final_key in net_final.params.iterkeys():
        ori_key = final_key
        if final_key in net_D_stage.params.iterkeys():
            merge(net_D_stage,ori_key,net_final,final_key)
            print('*'*10,final_key)
            pass
        elif final_key in net_C_stage.params.iterkeys():
            merge(net_C_stage,ori_key,net_final,final_key)
            print('-'*10,final_key)
            pass
    net_final.save(final_model)
