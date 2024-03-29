"""
This file is to generate training data for task2 & 3 in PNet
Will be merge with the previously generated data in another file:
PNet_Data.py

"""
import os
import random
from os.path import join, exists

import cv2
import numpy as np
import numpy.random as npr

from bbox_utils import getDataFromTxt, BBox
from gesture_utils import rotate, flip
from utils import IoU




def GenerateData(ftxt,data_path,net,augment=True):
    '''

    :param ftxt: name/path of the text file that contains image path,
                bounding box, and landmarks

    :param output: path of the output dir
    :param net: one of the net in the cascaded networks
    :param argument: apply augmentation or not
    :return:  images and related landmarks
    '''
    if net == "PNet":
        size = 12
    elif net == "RNet":
        size = 24
    elif net == "ONet":
        size = 48
    else:
        print('Net type error')
        return
    image_id = 0
    #
    f = open(join(OUTPUT,"gesture_%s_aug.txt" %(size)),'w')
    #dstdir = "train_landmark_few"
    # get image path , bounding box, and landmarks from file 'ftxt'
    data = getDataFromTxt(ftxt,data_path=data_path)
    idx = 0
    #image_path bbox landmark(5*2)
    for (imgPath, bbox, gestureGt) in data:
        #print imgPath
        F_imgs = []
        F_gesture = []
        #print(imgPath)
        #print(bbox)
        img = cv2.imread(imgPath)
        # cv2.imshow('img',img)
        # while cv2.waitKey(1) != ord('n'):
        #     continue
        assert(img is not None)
        img_h,img_w, _ = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        #get sub-image from bbox
        f_hand = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        # resize the gt image to specified size
        f_hand = cv2.resize(f_hand,(size,size))
        #initialize the gesture
        gesture = np.zeros(3)
        #print('gestureGt',gestureGt)
        for index, item in enumerate(gestureGt):
            #print('item',item)
            gesture[index] = item

        # #normalize gesture by dividing the width and height of the ground truth bounding box
        # # gestureGt is a list of tuples
        # for index, one in enumerate(gestureGt):
        #     # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
        #     rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
        #     # put the normalized value into the new list landmark
        #     gesture[index] = rv
        F_imgs.append(f_hand)
        F_gesture.append(gesture.reshape(3))
        #print('F_gesture',F_gesture)
        #gesture = np.zeros(3)
        if augment:
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, "images done")
            x1, y1, x2, y2 = gt_box
            #gt's width
            gt_w = x2 - x1 + 1
            #gt's height
            gt_h = y2 - y1 + 1        
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            #random shift
            for i in range(10):
                bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = int(max(x1+gt_w/2-bbox_size/2+delta_x,0))
                ny1 = int(max(y1+gt_h/2-bbox_size/2+delta_y,0))

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size

                if nx2 > img_w or ny2 > img_h:
                    continue

                crop_box = np.array([nx1,ny1,nx2,ny2])


                cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]
                resized_im = cv2.resize(cropped_im, (size, size))
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                if iou > 0.65:
                    F_imgs.append(resized_im)


                    for index, item in enumerate(gestureGt):
                        gesture[index] = item

                    # #normalize
                    # for index, one in enumerate(gestureGt):
                    #     rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                    #     gesture[index] = rv
                    F_gesture.append(gesture)
                    #gesture = np.zeros(3)
                    #gesture_ = F_gesture[-1].reshape(-1,2)
                    bbox = BBox([nx1,ny1,nx2,ny2])                    

                    #mirror                    
                    if random.choice([0,1]) > 0:
                        hand_flipped = flip(resized_im)
                        hand_flipped = cv2.resize(hand_flipped, (size, size))
                        #c*h*w
                        F_imgs.append(hand_flipped)
                        F_gesture.append(gesture)
                    #rotate
                    if random.choice([0,1]) > 0:
                        hand_rotated_by_alpha = rotate(img, bbox, 5)#anti-clockwise
                        #gesture_offset
                        #gesture_rotated = bbox.projectGesture(gesture_rotated)
                        hand_rotated_by_alpha = cv2.resize(hand_rotated_by_alpha, (size, size))
                        F_imgs.append(hand_rotated_by_alpha)
                        F_gesture.append(gesture)
                
                        #flip
                        hand_flipped = flip(hand_rotated_by_alpha)
                        hand_flipped = cv2.resize(hand_flipped, (size, size))
                        F_imgs.append(hand_flipped)
                        F_gesture.append(gesture)
                    
                    #anti-clockwise rotation
                    if random.choice([0,1]) > 0: 
                        hand_rotated_by_alpha = rotate(img, bbox, -5)#clockwise
                        #gesture_rotated = bbox.projectGesture(gesture_rotated)
                        hand_rotated_by_alpha = cv2.resize(hand_rotated_by_alpha, (size, size))
                        F_imgs.append(hand_rotated_by_alpha)
                        F_gesture.append(gesture)
                
                        hand_flipped = flip(hand_rotated_by_alpha)
                        hand_flipped = cv2.resize(hand_flipped, (size, size))
                        F_imgs.append(hand_flipped)
                        F_gesture.append(gesture)

            #print(len(F_imgs))
            #print(len(F_gesture))
            F_imgs = np.asarray(F_imgs)
            #print(F_imgs.shape)
            #print(F_gesture)
            F_gesture = np.asarray(F_gesture)
            #print F_imgs.shape
            #print F_landmarks.shape
            for i in range(len(F_imgs)):
                #if image_id % 100 == 0:

                    #print('image id : ', image_id)

                # if np.sum(np.where(F_gesture[i] <= 0, 1, 0)) > 0:
                #     continue
                #
                # if np.sum(np.where(F_gesture[i] >= 1, 1, 0)) > 0:
                #     continue

                name = join(dstdir,"%d.jpg" %(image_id))
                name.replace('\\','/')

                cv2.imwrite(name, F_imgs[i])
                gestures = map(str,list(F_gesture[i]))
                f.write(name+" -2 "+" ".join(gestures)+"\n")
                image_id = image_id + 1
            
    #print F_imgs.shape
    #print F_landmarks.shape
    #F_imgs = processImage(F_imgs)
    #shuffle_in_unison_scary(F_imgs, F_landmarks)
    
    f.close()
    return F_imgs,F_gesture

if __name__ == '__main__':
    dstdir = "../Dataset/Training/train_ONet_aug"
    OUTPUT = '../Dataset/Training'
    data_path = '../Dataset/Training'
    if not exists(OUTPUT):
        os.mkdir(OUTPUT)
    if not exists(dstdir):
        os.mkdir(dstdir)
    assert (exists(dstdir) and exists(OUTPUT))
    # train data
    net = "ONet"
    #the file contains the names of all the gesture training data
    train_txt = os.path.join('../Dataset/Training',"imglist_with_gesture.txt")
    imgs,gestures = GenerateData(train_txt,data_path,net,augment=True )
    
   
