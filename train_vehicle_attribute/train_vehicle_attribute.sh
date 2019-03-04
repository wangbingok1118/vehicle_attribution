export PYTHONPATH=/workspace/mnt/group/general-reg/wangbingbing/vehicle_workshop/vehicle_github_repo/train_vehicle_attribute_repo/python_layers:$PYTHONPATH
/opt/caffe/build/tools/caffe train \
-solver="/workspace/mnt/group/general-reg/wangbingbing/vehicle_workshop/vehicle_github_repo/train_vehicle_attribute_repo/model_net_files/0227/solver.prototxt" \
-weights="/workspace/mnt/group/general-reg/wangbingbing/vehicle_workshop/vehicle_github_repo/train_vehicle_attribute_repo/pretrainModels/mobilenet_v2.caffemodel" \
-gpu 3
