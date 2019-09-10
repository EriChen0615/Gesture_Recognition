#!/bin/bash
conda activate tensorflow-env
cd prepare_data
python PNet_gen_data.py --im_dir ../Dataset/Training --anno_file imglist_with_gesture.txt --save_dir ../Training_Data/-PNet-12
python gen_gesture.py --net PNet --im_dir ../Dataset/Training --anno_file imglist_with_gesture.txt --save_dir ../Training_Data/-PNet-12
python merge_gesture_and_data.py --net PNet --base_dir ../Training_Data/-PNet-12