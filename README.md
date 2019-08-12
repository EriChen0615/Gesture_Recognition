# Guide to generate the dataset
## Prerequistes:
opencv-python
numpy

## How to sample data
- run gen_img_dataset.py
- the program will prompt which gesture to adopt, place your hand inside the red bounding box and press 'space' to save the image.
- after certain number of shots, another the bounding box will change and so may the gesture. Follow instruction on the console.
- press 'q' to exit. 
- the program will automatically append to the image dataset. So there is no need for a one-off sampling.
- modify the DIR_NAME variable in gen_img_data to generate different datasets

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
- To add/delete a new gesture, simply add an item to the GS_BBOX_DICT. The value being the size of the bounding box .
- To build a separate dataset, change the DIR_NAME variable in gen_image_dataset.py and create an initial config.txt and respective directories with annotation.txt.
