import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import sys
sys.path.insert(0,'/opt/mxnet/python')
import mxnet as mx



def get_iterators(batch_size, data_shape=(3, 224, 112)):
#def get_iterators(batch_size, data_shape=(3, 128, 128)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = './data/train_dataset/didianwei-meituan-train-v4.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
       rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        path_imgrec         = './data/train_dataset/didianwei-meituan-val-v4.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
#    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
#    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')

    num_hidden =  2048
    #print("net",all_layers)
    net = mx.symbol.FullyConnected(data=net, num_hidden=2048, name='fc-1')
    net1 = mx.sym.Activation(net, act_type='relu')
    net2 = mx.symbol.Dropout(net1, p = 0.6)
    net_hidden = mx.symbol.FullyConnected(data=net2, num_hidden=num_hidden, name='fc-hidden')
    #net_gender_fc = mx.symbol.BlockGrad(net_gender_fc)
    net1 = mx.sym.Activation(net_hidden, act_type='relu')
    net_fc1 = mx.symbol.Dropout(net1, p = 0.6)
    net_fc = mx.symbol.FullyConnected(data=net_fc1, num_hidden=3, name='fc-waimai')
    net = mx.symbol.SoftmaxOutput(data=net_fc, name='softmax',grad_scale=0.1)


    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
model_prefix='./models/didianwei-waimaiv1'
checkpoint = mx.callback.do_checkpoint(model_prefix)
def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
        num_epoch=100,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='device',
        optimizer='sgd',
        epoch_end_callback=checkpoint,
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric),symbol,arg_params,aux_params

num_classes =3
batch_per_gpu = 128
num_gpus =4


batch_size = batch_per_gpu * num_gpus
(train, val) = get_iterators(batch_size)
sym, arg_params, aux_params = mx.model.load_checkpoint('./preTrainModels/resnet-18', 0)
(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
mod_score,modmy_symbol,modmy_arg_params,modmy_aux_params = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
