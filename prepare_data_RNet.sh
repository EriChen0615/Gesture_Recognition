#!/bin/bash

output_dir='Training_Data/RNet-24'
anno_name='imglist_with_gesture.txt'
model_prefix='Model/MTCNN/PNet'
model_prefix='Model/MTCNN/PNet'
model_epoch=30
raw_img_dir='Dataset/Training'

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf-gpu
cd prepare_data
python gen_data.py --test_mode PNet --anno_file $anno_name --im_dir ../$raw_img_dir --save_dir ../$output_dir --epoch $model_epoch --prefix ../$model_prefix
python gen_gesture.py --net RNet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir
python merge_gesture_and_data.py --net RNet --base_dir ../$output_dir
python gen_tfrecord.py --net RNet --data_dir ../$output_dir