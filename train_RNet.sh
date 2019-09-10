#!/bin/bash

model_name='MTCNN-test'
tfrecord_dir='Training_Data/RNet-24/imglists/RNet'
base_lr=0.1
end_epoch=30

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf-gpu

cd Train_Model
python train_PNet --model_name $model_name --tfrecord_dir $tfrecord_dir --base_lr $base_lr --end_epoch $end_epoch
