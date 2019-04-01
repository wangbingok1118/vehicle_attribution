export PYTHONPATH=/workspace/mnt/group/general-reg/wangbingbing/vehicle_workshop/vehicle_github_repo/vehicle_attribution/mix_caffe/D-stage-attribution/python_layer:$PYTHONPATHll
./caffe/build/tools/caffe train \
--solver=models/D_attribution_solver.prototxt \
--weights=models/D_attribution_train_pretrain.caffemodel \
--gpu=3
