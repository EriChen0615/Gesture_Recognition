"""
Can be used for both training and testing data
set mode to be 'Training' or 'Testing'

"""
from gen_img_dataset import load_config
import os

def append_field(wds,f):
    for w in f:
        wds.append(w)

if __name__ == '__main__':
    #define the mode to be testing or training
    mode = 'Testing'

    # definition of different gestures
    ONE_FIELD = '100'
    FIST_FIELD = '010'
    TWO_FIELD = '001'

    # configuration needs to be copied from gen_img_dataset.py if modified
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
    DIR_NAME = os.path.join('Dataset',mode)

    # COLLECTION SETTING
    SHOTS_PER_BOX = 5

    # GENERAL
    EXIT_KEY = 'q'

    config_path = os.path.join(DIR_NAME,'config.txt')
    total_counter, gest_counter, gest_path = load_config(config_path,GS_BBOX_DICT)
    gest_anno = {k:open(os.path.join(v,'annotation.txt'),'r') for (k,v) in gest_path.items()} # a dict of files



    with open(os.path.join('Dataset',mode,'imglist_with_gesture.txt'),'w+') as f_out:
        for gs,f_gs in gest_anno.items():
            annos = f_gs.read().splitlines()
            for anno in annos:
                words = anno.split(' ')
                words[0] = os.path.join(gs,words[0]) # change img_name to img_path as in Training directory

                if gs == 'one':
                    append_field(words,ONE_FIELD)
                elif gs == 'two':
                    append_field(words,TWO_FIELD)
                elif gs == 'fist':
                    append_field(words,FIST_FIELD)

                new_anno = ''
                for word in words:
                    new_anno += (word+' ')
                f_out.write(new_anno+'\n')

    for f in gest_anno.values():
        f.close()

