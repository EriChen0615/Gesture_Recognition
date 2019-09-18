#!/bin/bash

env_name=tensorflow_env
model_name=MTCNN-test
output_dir=Training_Data/$model_name
net_prefix=Model/MTCNN-test

raw_img_dir=Dataset/Training
anno_name=imglist_with_gesture.txt


tfrecord_dir=imglists
base_lr=0.1
pend_epoch=4
rend_epoch=4
oend_epoch=4

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $env_name

# RNet data generation
net=RNet
cd prepare_data
python gen_data.py --test_mode PNet --anno_file $anno_name --im_dir ../$raw_img_dir --save_dir ../$output_dir/$net --epoch $pend_epoch --prefix ../$net_prefix/PNet
python gen_gesture.py --net RNet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir/$net
python merge_gesture_and_data.py --net $net --base_dir ../$output_dir/$net
python gen_tfrecord.py --net RNet --data_dir ../$output_dir/$net
cd ..

echo 'RNet data generation completes!'


