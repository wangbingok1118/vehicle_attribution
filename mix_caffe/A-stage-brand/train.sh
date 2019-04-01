./caffe/build/tools/caffe train \
--solver=models/brand_solver.prototxt \
--weights=models/resnet18.v2.caffemodel \
--gpu=0,1,2,3