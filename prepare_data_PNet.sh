#!/bin/bash

# PNet data generation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf-gpu
cd prepare_data
python PNet_gen_data.py --im_dir ../Dataset/Training --anno_file imglist_with_gesture.txt --save_dir ../Training_Data/PNet-12
python gen_gesture.py --net PNet --im_dir ../Dataset/Training --anno_file imglist_with_gesture.txt --save_dir ../Training_Data/PNet-12
python merge_gesture_and_data.py --net PNet --base_dir ../Training_Data/PNet-12
python gen_tfrecord.py --net PNet --data_dir ../Training_Data/PNet-12

