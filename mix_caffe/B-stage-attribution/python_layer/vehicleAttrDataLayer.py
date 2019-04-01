import sys
import cv2
import re
sys.path.append('/opt/caffe/python')
import caffe
import numpy as np
import random
import os

def centerCrop(im, crop_size):
    h,w = im.shape[:2]
    h_crop = crop_size[0]
    w_crop = crop_size[1]
    h_off,w_off=(h-h_crop)//2,(w-w_crop)//2
    im = im[h_off:h_off+h_crop, w_off:w_off+w_crop]
    return im

def RondomRotate(image, max_angle):
#   image = cv2.imread(image)
    h,w,_ = image.shape
    c_x =  w / 2
    c_y =  h / 2
    angle = np.random.randint(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
    roate_img = cv2.warpAffine(image, M, (w, h))
    return roate_img

def mirror(img):
    img = cv2.flip(img, 1)
    return img


def illumination(img):
    r0 = random.uniform(0.0, 1.0)
    r1 = random.uniform(0.0, 1.0)
    hue = r0 * 0.1 + 1
    exposure = r1 * 0.5 + 1
    if random.uniform(0.0, 1.0) > 0.5:
        exposure = 1.0 / exposure
    if random.uniform(0.0, 1.0) > 0.5:
        hue = 1.0 / hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] * exposure
    v = np.clip(v, 0, 255)
    # v = v.astype(np.int32)
    h = hsv[:, :, 0] * hue
    h = np.clip(h, 0, 255)
    # h = h.astype(np.int32)

    hsv[:, :, 2] = v.astype(np.uint8)
    hsv[:, :, 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# read img list
def readSrcFile(root_folder=None,source=None):
    imgPathLabelList = []
    with open(source,'r') as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imagePath = temp[0] if not root_folder  else os.path.join(root_folder,temp[0])
            labelList = [int(i) for i in temp[1:]]
            imgPathLabelList.append([imagePath, labelList])
    return imgPathLabelList

################################################################################
#########################Train Data Layer By Python#############################
################################################################################
class Data_Layer_train(caffe.Layer):

    def setup(self, bottom, top):
#	print 'bottom is:',bottom
        if len(top) != 4: # 3 (attribute numbers)  + 1
            raise Exception("Need to define tops (data, label)")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")
        params = eval(self.param_str)
        self.max_rotate_angle = 30 # defaule
        self.transform_params = params['transform_param']
        self.crop_size = self.transform_params['crop_size'] # tuple (168,168)
        self.mirror = None if 'mirror' not in self.transform_params else self.transform_params['mirror']
        self.image_data_param = params['image_data_param']
        self.batch_size = self.image_data_param['batch_size']
        self.new_image_size = None if 'new_image_size' not in self.image_data_param  else self.image_data_param['new_image_size'] # tuple (192,192)
        if 'root_folder' in self.image_data_param:
            self.imgLabelList = readSrcFile(root_folder=self.image_data_param['root_folder'],\
            source=self.image_data_param['source'])
        else:
            self.imgLabelList = readSrcFile(root_folder=None, \
                                    source=self.image_data_param['source'])
        self._cur = 0  # use this to check if we need to restart the list of images
        self.data_aug_type = ["normal"]
        if self.mirror == None:
            self.data_aug_type.append("mirror")
        top[0].reshape(self.batch_size, 3, self.crop_size[0], self.crop_size[1])
        for i in xrange(1, 4):
            top[i].reshape(self.batch_size, 1)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            im, labelList = self.load_next_image()
            top[0].data[itt, ...] = im
            for nums in xrange(3):
                top[nums+1].data[itt, ...] = labelList[nums]


    def backward(self, top, propagate_down, bottom):
        pass


    def load_next_image(self):
        # If we have finished forwarding all images, then an epoch has finished
        # and it is time to start a new one
        if self._cur == len(self.imgLabelList):
            self._cur = 0
        if self._cur == 0:
            random.shuffle(self.imgLabelList)
        img_path, labelList = self.imgLabelList[self._cur]
        self._cur += 1
        # bgr
        image = cv2.imread(img_path)
        image = image.astype(np.float32)
        if self.new_image_size:
            image = cv2.resize(image,self.new_image_size)
        image = self.data_augment(image)
        # normalization
        image = image.transpose((2, 0, 1))
        if 'mean_value' in self.transform_params:
            mean = self.transform_params['mean_value'] # list
            for i in range(3):
                image[i,:,:] += mean[i]
        if 'scale' in self.transform_params:
            scale = self.transform_params['scale']
            image *= scale
        return image, labelList

    # mirror, illumination, mirror+illumination
    def data_augment(self, image):
        image = RondomRotate(image,self.max_rotate_angle)
        image = centerCrop(image,self.crop_size)
        # choose a type of data augment
        idx = random.randint(0, len(self.data_aug_type) - 1)
        if self.data_aug_type[idx] == 'mirror':
            image = mirror(image)
        elif self.data_aug_type[idx] == 'illumination':
            image = illumination(image)
        elif self.data_aug_type[idx] == 'mirror_illumination':
            image = illumination(image)
            image = mirror(image)
        else:
            image = image
        return image

