"""
copy  B stage trained model to D stage pretrain model
conv1 conv2 conv3   cov4 conv 5 : copy B stage trained model
"""
import numpy as np
import sys, os

sys.path.insert(0, './caffe/python')
import caffe


B_stage_proto = './models/B_stage_attribution_train.prototxt'
B_stage_model = 'models/B_stage_attribution_final.caffemodel'

final_proto = './models/D_attribution_train.prototxt'
final_model = './models/D_attribution_train_pretrain.caffemodel'

def merge(net_ori,ori_key, net_final, final_key):
    weights = net_ori.params[ori_key]
    for i , w in enumerate(weights):
        net_final.params[final_key][i].data[...] = w.data

    pass

if __name__ == '__main__':
    net_B_stage = caffe.Net(B_stage_proto, B_stage_model, caffe.TRAIN)
    net_final = caffe.Net(final_proto, caffe.TEST)
    for final_key in net_final.params.iterkeys():
        ori_key = final_key.split('_attribution')[0]
        merge(net_B_stage, ori_key, net_final, final_key)
        print(ori_key, final_key)
    net_final.save(final_model)
