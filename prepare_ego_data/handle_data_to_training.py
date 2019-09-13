import os, os.path

def load_config(filename,gs_bbox_dict):
    t_count = 0
    dir_count = {}
    dir_p = {}
    for gs in gs_bbox_dict.keys():
        dir_count[gs] = 0
    for gs in gs_bbox_dict.keys():
        dir_p[gs] = None

    with open(filename) as f:
        contents = f.read().splitlines()
        #print(contents)
        for line in contents:
            words = line.split()
            #print(words)
            if words[0]=='@': t_count = int(words[1])
            elif words[0]=='!': dir_count[words[2]]=int(words[1])
            elif words[0]=='-': dir_p[words[2]]=words[1]
    return t_count, dir_count, dir_p

def append_field(wds,f):
    for w in f:
        wds.append(w)

if __name__ == '__main__':
    # definition of different gestures
    # ONE_FIELD = '100'
    # FIST_FIELD = '010'
    # TWO_FIELD = '001'

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
    GS_BBOX_DICT = {'SingleBad':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC),
                    'SingleGood':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*SHORT_FAC),
                    'SingleOne':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC),
                    'SingleTwo':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC),
                    'SingleFour':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC),
                    'SingleSix':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC),
                    'SingleEight':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC),
                    'SingleNine':(WINDOW_WIDTH*SHORT_FAC,WINDOW_HEIGHT*LONG_FAC)}

    COLLECT_LABEL = ['SingleBad','SingleGood','SingleOne','SingleTwo','SingleFour','SingleSix',
                     'SingleEight', 'SingleNine'] # the label to collect

    # DATA PATH
    DIR_NAME = os.path.join('../ego_data','Training')

    # COLLECTION SETTING
    SHOTS_PER_BOX = 5

    # GENERAL
    EXIT_KEY = 'q'

    config_path = os.path.join(DIR_NAME,'config.txt')
    total_counter, gest_counter, gest_path = load_config(config_path,GS_BBOX_DICT)
    gest_anno = {k:open(os.path.join(v,'annotation.txt'),'r') for (k,v) in gest_path.items()} # a dict of files



    with open(os.path.join('../ego_data','Training','imglist_without_gesture.txt'),'w+') as f_out:
        for gs,f_gs in gest_anno.items():
            annos = f_gs.read().splitlines()
            for anno in annos:
                words = anno.split(' ')
                words[0] = os.path.join(gs,words[0]) # change img_name to img_path as in Training directory

                # if gs == 'one':
                #     append_field(words,ONE_FIELD)
                # elif gs == 'two':
                #     append_field(words,TWO_FIELD)
                # elif gs == 'fist':
                #     append_field(words,FIST_FIELD)

                new_anno = ''
                for word in words:
                    new_anno += (word+' ')
                f_out.write(new_anno+'\n')

    for f in gest_anno.values():
        f.close()
