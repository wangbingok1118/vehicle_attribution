#coding=utf-8
import argparse
import numpy as np
import sys
sys.path.insert(0,'./caffe/python')
import caffe


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()

def load_labelName(file):
    label_dict = dict()
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            label_dict[int(line[0])] = line[1]
    return label_dict

def eval():
    nh, nw = 168, 168
    img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)

    caffe.set_mode_cpu()
    net = caffe.Net(args.proto, args.model, caffe.TEST)

    im = caffe.io.load_image(args.image)
    h, w, _ = im.shape
    if h < w:
        off = (w - h) / 2
        im = im[:, off:off + h]
    else:
        off = (h - w) / 2
        im = im[off:off + h, :]
    im = caffe.io.resize_image(im, [nh, nw])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    transformer.set_mean('data', img_mean)
    transformer.set_input_scale('data', 0.017)

    net.blobs['data'].reshape(1, 3, nh, nw)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    prob_brand = out['prob-brand']
    prob_brand = np.squeeze(prob_brand)
    idx_brand = np.argsort(-prob_brand)

    prob_type = out['prob-type']
    prob_type = np.squeeze(prob_type)
    idx_type = np.argsort(-prob_type)

    prob_direction = out['prob-direction']
    prob_direction = np.squeeze(prob_direction)
    idx_direction = np.argsort(-prob_direction)

    prob_color = out['prob-color']
    prob_color = np.squeeze(prob_color)
    idx_color = np.argsort(-prob_color)

    feature = out['vehicle-feature']
    print(feature.shape)
    print(np.resize(feature,512))

    #label_names = np.loadtxt('synset.txt', str, delimiter='\t')
    brand_label = load_labelName('brand_label.txt')
    type_label = load_labelName('type_label.txt')
    direction_label = load_labelName('direction_label.txt')
    color_label = load_labelName('color_label.txt')
    for i in range(1):
        label_brand = idx_brand[i]
        label_type = idx_type[i]
        label_direction = idx_direction[i]
        label_color = idx_color[i]
        print('%.2f - %s  - %s' % (prob_brand[label_brand],str(label_brand) ,brand_label[label_brand]))
        print('%.2f - %s  - %s' % (prob_type[label_type], str(label_type), type_label[label_type]))
        print('%.2f - %s  - %s' % (prob_direction[label_direction], str(label_direction), direction_label[label_direction]))
        print('%.2f - %s  - %s' % (prob_color[label_color], str(label_color), color_label[label_color]))
    return


if __name__ == '__main__':
    eval()