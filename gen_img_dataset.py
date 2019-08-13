import cv2
import numpy as np
from time import sleep
import random
import os

def drawBoxes(im, box,rgb=(0,255,0)):
    _x1 = box[0]
    _y1 = box[1]
    _x2 = box[2]
    _y2 = box[3]
    cv2.rectangle(im, (int(_x1), int(_y1)), (int(_x2), int(_y2)), rgb, 1)
    return im

def pause():
    while cv2.waitKey(1)!=ord('n'):
        continue

def get_next_gs_bbox(gs_bbox_dict,_window,collect_label):
    _gest = None
    while not (_gest in collect_label) or _gest is None:
        _gest, (w,h) = random.choice(list(gs_bbox_dict.items()))
    _x_ratio = np.random.normal(loc=0.5,scale=0.15)
    _y_ratio = np.random.normal(loc=0.5,scale=0.15)
    w_r = np.random.normal(loc=1,scale=0.05)
    h_r = np.random.normal(loc=1,scale=0.05)

    w = w*w_r
    h = h*h_r

    _x_center = WINDOW_WIDTH * _x_ratio + _window[0]
    _y_center = WINDOW_HEIGHT * _y_ratio + _window[1]
    _x1 = _x_center - w/2
    _x2 = _x_center + w/2
    _y1 = _y_center - h/2
    _y2 = _y_center + h/2
    # make sure the bounding box is contained in the window
    _x1 = max(_x1,window[0])
    _x2 = min(_x2,window[2])
    _y1 = max(_y1,window[1])
    _y2 = min(_y2,window[3])

    bbox = [_x1,_y1,_x2,_y2]
    return _gest, bbox

def load_config(filename):
    t_count = 0
    dir_count = {}
    dir_p = {}
    for gs in GS_BBOX_DICT.keys():
        dir_count[gs] = 0
    for gs in GS_BBOX_DICT.keys():
        dir_p[gs] = None

    path = os.path.join(DIR_NAME,filename)
    with open(path) as f:
        contents = f.read().splitlines()
        print(contents)
        for line in contents:
            words = line.split()
            #print(words)
            if words[0]=='@': t_count = int(words[1])
            elif words[0]=='!': dir_count[words[2]]=int(words[1])
            elif words[0]=='-': dir_p[words[2]]=words[1]
    return t_count, dir_count, dir_p

def export_config(filename,t_count,dir_count,dir_p):
    with open(os.path.join(DIR_NAME,filename),'w+') as f:
        f.write('@ {0}\n'.format(t_count))
        for gs,num in dir_count.items(): # write categorical count
            f.write('! {0} {1}\n'.format(num,gs))
        for gs,path in dir_p.items(): # write path
            f.write('- {0} {1}\n'.format(path,gs))

if __name__ == '__main__':

    # --------- CONFIGURATION STARTS --------- #

    # WINDOW SETTING
    VIDEO_WIDTH = 960
    VIDEO_HEIGHT = 540

    # WINDOW SETTING
    WINDOW_WIDTH = 240
    WINDOW_HEIGHT = 240

    # BOUNDING BOX SETTING
    LONG_FAC = 0.6
    SHORT_FAC = 0.4
    GS_BBOX_DICT = {'one':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC),
                    'fist':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*SHORT_FAC),
                    'two':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC)}
    COLLECT_LABEL = ['one','fist','two'] # the label to collect

    # DATA PATH
    DIR_NAME = os.path.join('Dataset','Training')

    # COLLECTION SETTING
    SHOTS_PER_BOX = 5

    # GENERAL
    EXIT_KEY = 'q'

    # --------- CONFIGURATION ENDS ---------- #

    # --------- FILE I/O READING -------- #
    total_counter, gest_counter, gest_path = load_config('config.txt')

    gest_anno = {k:open(os.path.join(v,'annotation.txt'),'a+') for (k,v) in gest_path.items()} # a dict of files

    # print(total_counter)
    # print(gest_counter)
    # print(gest_path)
    # print(gest_anno)

    
    
    
    # ----------- DATA COLLECTION ----------- #
    cam = cv2.VideoCapture(0)
    cam.set(3,VIDEO_WIDTH)
    cam.set(4,VIDEO_HEIGHT)


    y = np.random.normal(loc=0.5,scale=0.1,size=1000)
    # x = np.arange(len(y))
    # plt.hist(y)
    # plt.show()

    shot_counter = -1 # indicates that bbox needs to be refreshed
    bounding_box = [0,0,0,0]
    while True:
        __, frame = cam.read()
        frame = cv2.flip(frame,1)
        img = frame.copy()
        if shot_counter == -1:
            # Get the next window (normally distributed)
            x_ratio = np.random.normal(loc=0.5,scale=0.1)
            y_ratio = np.random.normal(loc=0.5,scale=0.1)

            x_center = VIDEO_WIDTH * x_ratio
            y_center = VIDEO_HEIGHT * y_ratio
            x1 = x_center - WINDOW_WIDTH/2
            x2 = x_center + WINDOW_WIDTH/2
            y1 = y_center - WINDOW_HEIGHT/2
            y2 = y_center + WINDOW_HEIGHT/2
            # in case of overflow
            x1 = max(x1,0)
            x2 = min(x2,VIDEO_WIDTH)
            y1 = max(y1,0)
            y2 = min(y2,VIDEO_HEIGHT)
            window = [x1,y1,x2,y2]
            # update the next gesture and bbox size
            gest, bounding_box = get_next_gs_bbox(GS_BBOX_DICT,window,COLLECT_LABEL)

            shot_counter = 0
            #print('Window: {0}\nBoundingbox: {1}'.format(window,bounding_box))
            print('>>>>>>>>Recording gesture {0}<<<<<<<<'.format(gest.upper()))
            print('Please gesture in the red bounding box')
        drawBoxes(frame,window)
        drawBoxes(frame,bounding_box,(0,0,255))
        cv2.imshow('camera',frame)

        if cv2.waitKey(1)== 0x20: # press space to take shot

            img_name = '{0}_{1}.jpg'.format(gest,gest_counter[gest])
            # annotate image and save
            gt_bbox = [int(bounding_box[0]-window[0]),int(bounding_box[1]-window[1]),int(bounding_box[2]-window[0]),int(bounding_box[3]-window[1])]
            im_save = img[int(window[1]):int(window[3]),int(window[0]):int(window[2])]

            cv2.imwrite(os.path.join(gest_path[gest],img_name),im_save)
            gest_anno[gest].write(img_name+' '+str(gt_bbox[0])+' '+str(gt_bbox[1])+' '+str(gt_bbox[2])+' '+str(gt_bbox[3])+'\n')  # append to annotation

            # display saved image
            im_show = im_save.copy()
            im_show = drawBoxes(im_show,gt_bbox)
            cv2.imshow('saved_image',im_show)

            # increment counters
            gest_counter[gest] += 1
            total_counter += 1
            shot_counter += 1

            # display
            print('-------------------------------------')
            print('{0} is saved with bbox {1}'.format(img_name,gt_bbox))
            print('SHOT: {0} / {1}'.format(shot_counter,SHOTS_PER_BOX))

        # if bounding box is restricted, press b to abandon this set
        elif cv2.waitKey(2)== ord('b'):
            shot_counter=-1

        if shot_counter == SHOTS_PER_BOX:
            shot_counter = -1

        if cv2.waitKey(1)==ord('q'):
            break


    # --------- FILE I/O WRITING  -------- #
    export_config('config.txt',total_counter,gest_counter,gest_path)

    for f in gest_anno.values():
        f.close()

