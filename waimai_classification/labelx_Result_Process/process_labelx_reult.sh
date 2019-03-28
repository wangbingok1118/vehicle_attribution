#!/usr/bin/env bash
#set -x
labelFile=$1
savePath=$2
# download all images in labelx
qshellLogin-AtVideo.sh
for i_data in `cat $labelFile | jq -r '.url'| cut -d '/' -f 6| sort -u`
do
    echo $i_data
    dest_dir=$i_data
    mkdir $dest_dir
    prefix="Classification\/waimai-crop\/"$i_data
    sed  "s/dest_dir_variabel/${dest_dir}/g" qdownload.conf > $i_data.conf
    sed -i "s/prefix_variable/${prefix}/g" $i_data.conf
    qshell qdownload -c 20 $i_data.conf
#    rm download.log
done
# split images in sub_class folder
while read line;do
    url=`echo $line |jq -r '.url'`
    i_data=`echo $url | cut -d '/' -f 6`
    i_path=`echo $url | cut -c '34-' `
    local_image_path=$i_data/$i_path
    class=`echo $line|jq -r '.label[0].data[0].class'|xargs`
    save_image_path=$savePath"/"$class
    if [ ! -d $save_image_path ];then
        mkdir $save_image_path
    fi
    cp $local_image_path $save_image_path
done < $labelFile
