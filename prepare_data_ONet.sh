#!/bin/bash

output_dir='Training_Data/ONet-48'
anno_name='imglist_with_gesture.txt'
pnet_prefix='Model/MTCNN/PNet'
rnet_prefix='Model/MTCNN/RNet'
pnet_epoch=30
rnet_epoch=30
raw_img_dir='Dataset/Training'

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf-gpu
cd prepare_data
python gen_data.py --test_mode RNet --anno_file $anno_name --im_dir ../$raw_img_dir --save_dir ../$output_dir --epoch $pnet_epoch $rnet_epoch --prefix ../$pnet_prefix ../$rnet_prefix
python gen_gesture.py --net ONet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir
python merge_gesture_and_data.py --net ONet --base_dir ../$output_dir
python gen_tfrecord.py --net ONet --data_dir ../$output_dir