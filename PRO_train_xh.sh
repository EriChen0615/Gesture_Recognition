#!/bin/bash

env_name=tf-gpu
model_name=MTCNN-test
output_dir=ego_data/Training
net_prefix=Model/$model_name

raw_img_dir=ego_data/Training
anno_name=imglist_without_gesture.txt


tfrecord_dir=imglists
p_base_lr=0.1
r_base_lr=0.01
o_base_lr=0.01 #haven't experiment yet
pend_epoch=30
rend_epoch=22
oend_epoch=22


# PNet data generation
net=PNet
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $env_name
# source ~/venv/bin/activate
cd prepare_ego_data
python PNet_gen_data.py --im_dir ../$raw_img_dir  --save_dir ../$output_dir/$net
python handle_data_to_training.py 
python gen_gesture.py --net PNet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir/$net
python merge_gesture_and_data.py --net $net --base_dir ../$output_dir/$net
python gen_tfrecord.py --net PNet --data_dir ../$output_dir/$net
cd ..
echo 'PNet data generation completes!'

# PNet training

cd Train_Model
python train_net.py --net $net --model_name $model_name --tfrecord_dir ../$output_dir/$net/$tfrecord_dir --base_lr $p_base_lr --end_epoch $pend_epoch
cd ..

echo 'PNet training completes!'

# # RNet data generation
# net=RNet
# cd prepare_data
# python gen_data.py --test_mode PNet --anno_file $anno_name --im_dir ../$raw_img_dir --save_dir ../$output_dir/$net --epoch $pend_epoch --prefix ../$net_prefix/PNet
# python gen_gesture.py --net RNet --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir/$net
# python merge_gesture_and_data.py --net $net --base_dir ../$output_dir/$net
# python gen_tfrecord.py --net RNet --data_dir ../$output_dir/$net
# cd ..

# echo 'RNet data generation completes!'

# # RNet training
# cd Train_Model
# python train_net.py --net $net --model_name $model_name --tfrecord_dir ../$output_dir/$net/$tfrecord_dir --r_base_lr $base_lr --end_epoch $rend_epoch
# cd ..

# echo 'RNet training completes'

# # ONet data generation
# net=ONet
# cd prepare_data
# python gen_data.py --test_mode RNet --anno_file $anno_name --im_dir ../$raw_img_dir --save_dir ../$output_dir/$net --epoch $pend_epoch $rend_epoch --prefix ../$net_prefix/PNet ../$net_prefix/RNet
# python gen_gesture.py --net $net --im_dir ../$raw_img_dir --anno_file $anno_name --save_dir ../$output_dir/$net
# python merge_gesture_and_data.py --net $net --base_dir ../$output_dir/$net
# python gen_tfrecord.py --net $net --data_dir ../$output_dir/$net
# cd ..

# echo 'ONet data generation completes'

# # ONet training
# cd Train_Model
# python train_net.py --net $net --model_name $model_name --tfrecord_dir ../$output_dir/$net/$tfrecord_dir --o_base_lr $base_lr --end_epoch $oend_epoch
# cd ..

# echo 'ONet training completes'

# echo 'PRO training completes!'
# echo "model name: ${model_name}"
# echo "output_dir: ${output_dir}"
# echo "save_path: ${net_prefix}"
# echo "dataset: ${raw_img_dir}"
