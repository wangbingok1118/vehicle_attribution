"""
copy A stage trained model and B stage trained model to C stage pretrain model
conv1 conv2 conv3  : copy B stage trained model
conv4 conv5 : copy A stage trained model
"""
import numpy as np
import sys, os

sys.path.insert(0, './caffe/python')
import caffe

A_stage_proto = './models/A_stage_brand_train.prototxt'
A_stage_model = 'models/A-stage-brand_iter_50000.caffemodel'

B_stage_proto = './models/B_stage_attribution_train.prototxt'
B_stage_model = 'models/B-stage-attribution_iter_20000.caffemodel'

final_proto = './models/C_brand_train.prototxt'
final_model = './models/C_brand_train_pretrain.caffemodel'

def merge(net_ori,ori_key, net_final, final_key):
    weights = net_ori.params[ori_key]
    for i , w in enumerate(weights):
        net_final.params[final_key][i].data[...] = w.data

    pass

if __name__ == '__main__':
    net_A_stage = caffe.Net(A_stage_proto, A_stage_model, caffe.TRAIN)
    net_B_stage = caffe.Net(B_stage_proto, B_stage_model, caffe.TRAIN)
    net_final = caffe.Net(final_proto, caffe.TEST)
    for final_key in net_final.params.iterkeys():
        if 'brand' in final_key: # conv4 cov5 , copy A stage weight to final model
            ori_key = final_key.split('_brand')[0]
            merge(net_A_stage, ori_key, net_final, final_key)
            print('*'*10)
            print(ori_key, final_key)
            pass
        else: # conv1 conv2 conv3 , copy B stage weight to final model
            ori_key = final_key
            merge(net_B_stage, ori_key, net_final, final_key)
            print('-' * 10)
            print(ori_key, final_key)

    net_final.save(final_model)
