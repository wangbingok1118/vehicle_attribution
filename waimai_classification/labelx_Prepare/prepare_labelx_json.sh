#!/usr/bin/env bash
data_flag=$1
bucket_name="xuhui-waimai-collect-crop"
bucket_prefix="Classification/waimai-crop/"$data_flag

qshellLogin-AtVideo.sh
if [ ! -d $data_flag ];then
    mkdir $data_flag
fi
qshell listbucket2 $bucket_name -p $bucket_prefix > $data_flag/xuhui_waimai_$data_flag.log
cat $data_flag/xuhui_waimai_$data_flag.log | awk '{print "http://po1gsb6b3.bkt.clouddn.com/"$1}' > $data_flag/xuhui_waimai_$data_flag.url
python generate_labelx_json.py $data_flag/xuhui_waimai_$data_flag.url
rm $data_flag/xuhui_waimai_$data_flag.log
rm $data_flag/xuhui_waimai_$data_flag.url
