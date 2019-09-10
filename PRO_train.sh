#!/bin/bash

env_name='tf-gpu'
output_dir='Training_Data'
net_prefix='Model/MTCNN-test/PNet'

raw_img_dir='Dataset/Training'
anno_name='imglist_with_gesture.txt'

pnet_prefix='Model/MTCNN-test/PNet'


rnet_prefix='Model/MTCNN-test/RNet'


model_name='MTCNN-test'
tfrecord_dir='Training_Data/PNet-12/imglists'
base_lr=0.1
pend_epoch=5
rend_epoch=5
oend_epoch=5


# PNet data generation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $env_name
cd prepare_data
python PNet_gen_data.py --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$poutput_dir
python gen_gesture.py --net PNet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$poutput_dir
python merge_gesture_and_data.py --net PNet --base_dir ../$poutput_dir
python gen_tfrecord.py --net PNet --data_dir ../$poutput_dir

echo('PNet data generation completes!')

# PNet training

cd ..
cd Train_Model
python train_net.py --net $net --model_name $model_name --tfrecord_dir ../$tfrecord_dir/PNet --base_lr $base_lr --end_epoch $end_epoch
cd ..

# RNet data generation
python gen_data.py --test_mode PNet --anno_file $anno_name --im_dir ../$raw_img_dir --save_dir ../$output_dir --epoch $pend_epoch --prefix ../$model_prefix
python gen_gesture.py --net RNet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir
python merge_gesture_and_data.py --net RNet --base_dir ../$output_dir
python gen_tfrecord.py --net RNet --data_dir ../$output_dir

# ONet data generation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf-gpu
cd prepare_data
python gen_data.py --test_mode RNet --anno_file $anno_name --im_dir ../$raw_img_dir --save_dir ../$output_dir --epoch $pnet_epoch $rnet_epoch --prefix ../$pnet_prefix ../$rnet_prefix
python gen_gesture.py --net ONet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir
python merge_gesture_and_data.py --net ONet --base_dir ../$output_dir
python gen_tfrecord.py --net ONet --data_dir ../$output_dir