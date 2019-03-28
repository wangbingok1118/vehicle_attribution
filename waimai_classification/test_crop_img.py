import mxnet as mx
import numpy as np
import cv2
import os
import time
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])
# img_path_root='/workspace/mnt/group/video-det/zhangfeiyun/projects/fashion/data'
# imsave_root='/workspace/mnt/group/video-det/zhangfeiyun/projects/fashion/data/results/'
test_txt = '/workspace/mnt/group/video-det/zhangfeiyun/projects/fashion/waimai/0311/test_0311.txt'
imsave_root = '/workspace/mnt/group/video-det/zhangfeiyun/projects/fashion/waimai/0311/out'

input_shape = [1, 3, 224, 112]
ctx = mx.gpu()
sym, arg_params, aux_params = mx.model.load_checkpoint(
    '/workspace/mnt/group/video-det/zhangfeiyun/projects/fashion/waimai/model_didianwei/didianwei-waimaiv1', 39)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
data_input = [mx.io.DataDesc(str('data'), input_shape, layout='NCHW')]
# mod.infer_shape(data_input)
mod.bind(data_input, for_training=False)

mod.set_params(arg_params, aux_params, allow_missing=True)
# labels =['alamira_hijab','arab','burka','casual','chador','uniform','xinjiang']
labels = ['meituan', 'ele', 'other']
with open(test_txt) as f:
    for line in f.readlines():
        time_start = time.time()
        line = line.strip('\n')
        # img_path  = os.path.join(img_path_root,line)
        im = cv2.imread(line)
        img = cv2.resize(im, (112, 224))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        mod.forward(Batch([mx.nd.array(img)]))
        data = []
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        a = np.argsort(prob)[::-1]
        for idx, output in enumerate(mod.get_outputs()):
            print(labels[idx], output)
            temp = output.asnumpy()
            max_idx = np.argmax(temp)
            # label_name = label_dicts[label_names[idx]][max_idx]
            score = temp[0, max_idx]
            if len(temp[0, :]) > 2:
                score *= len(temp[0, :])

            print(labels[idx] + ' : ' + label_dicts[labels[idx]][max_idx])
            # data.append({"class":label_dicts[label_names[idx]][max_idx],"score": str(score)})

    #    imsave_name = os.path.basename(line)
    #     if prob[a[0]]>0:
    #         if labels[a[0]] == 'meituan':
    #             imsave_path = os.path.join(imsave_root,labels[a[0]],imsave_name)
    #             cv2.imwrite(imsave_path,im)
# elif labels[a[0]] == 'ele':
# 	imsave_path = os.path.join(imsave_root,labels[a[0]],imsave_name)
#             cv2.imwrite(imsave_path,im)
# elif abels[a[0]] == 'other':

#             imsave_path = os.path.join(imsave_root,labels[a[0]],imsave_name)
#             cv2.imwrite(imsave_path,im)







