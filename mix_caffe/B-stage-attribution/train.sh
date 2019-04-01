export PYTHONPATH=/workspace/mnt/group/general-reg/wangbingbing/vehicle_workshop/vehicle_github_repo/vehicle_attribution/mix_caffe/B-stage-attribution/python_layer:$PYTHONPATHll
./caffe/build/tools/caffe train \
--solver=models/attribution_solver.prototxt \
--weights=models/A-stage-brand_iter_5000.caffemodel \
--gpu=3