./caffe/build/tools/caffe train \
--solver=models/C_brand_solver.prototxt \
--weights=models/C_brand_train_pretrain.caffemodel \
--gpu=0,1,2,3