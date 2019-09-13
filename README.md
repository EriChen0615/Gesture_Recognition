# Guide to generate the dataset
## Prerequistes:
- opencv-python
- numpy

## How to sample data
- run gen_img_dataset.py
- the program will prompt which gesture to adopt, place your hand inside the red bounding box and press 'space' to save the image.
- if the bounding box position is not desirable, press 'b' to regenerate bounding box
- after certain number of shots, another the bounding box will change and so may the gesture. Follow instruction on the console.
- press 'q' to exit. 
- the program will automatically append to the image dataset. So there is no need for a one-off sampling.


## Customize sampling
- modify the ```DIR_NAME``` variable in gen_img_data.py to generate different datasets
- modify ```VIDEO_WIDTH VIDEO_HEIGH TWINDOW_WIDTH WINDOW_HEIGHT``` in gen_img_data.py to adjust resolution 
- modify ```SHOTS_PER_BOX``` to modify number of shots before bounding box changes
- add gesture name in ```COLLECT_LABEL``` to enable collection

## Data format
- gestures are stored by their category in format <gesture>_<number>.jpg
- annotations are stored in annotation.txt in respective directory. In format <img_path> <x1> <y1> <x2> <y2>

## Configure the dataset
- To reinitialize the dataset, make sure the config.txt file is as follows:
```
@ 0
! 0 one
! 0 fist
! 0 <your_gesture>
- Dataset\one/ one
- Dataset\fist/ fist
- <gesture_dir> <gesture>
```
- To add/delete a new gesture, simply add an item to the ```GS_BBOX_DICT```. The value being the size of the bounding box .
- To build a separate dataset, change the ```DIR_NAME``` variable in gen_image_dataset.py and create an initial config.txt and respective directories with annotation.txt.


# Guide to generate the data for Training
## Prerequistes:
- os
- cv2

# Data Generation and Training
## PRO_train
PRO_train.bash is a helper program which helps you generate training data and train the nets in one go. Simply run ```bash PRO_train.sh``` to train your network. To configure the bash file:
- ```env_name``` anaconda virtual environment name where tensorflow is installed
- ```model_name``` the name of the model
- ```output_dir``` the directory where the training data will be output
- ```net_prefix``` the prefix where the net is saved
- ```raw_img_dir``` the base directory of the dataset used
- ```anno_name``` the name of the annotation file of the raw dataset under its base directory as specified in ```raw_img_dir```
- ```tfrecord_dir``` sub dir to save tfrecord
- ```base_lr``` base learn rate for P R ONet training
- ```pend_epoch```,```rend_epoch```,```oend_epoch``` epoch number to be trained on P R ONet
