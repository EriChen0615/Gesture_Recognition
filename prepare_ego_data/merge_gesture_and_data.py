import numpy as np
import numpy.random as npr
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate gesture data for training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net',dest='net',help='a net in [PNet, RNet, ONet]')
    parser.add_argument('--base_dir',dest='base_dir',help='base directory for the training data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.base_dir
    net = args.net
    assert(net in ['PNet','RNet','ONet'])
    #anno_file = os.path.join(data_dir, "annotation.txt")

    if net == 'PNet':
        size = 12
    elif net == 'RNet':
        size = 24
    elif net == 'ONet':
        size = 48

    with open(os.path.join(data_dir, 'pos_%s.txt' %(size)), 'r') as f:
        pos = f.readlines()

    with open(os.path.join(data_dir, 'neg_%s.txt' %(size)), 'r') as f:
        neg = f.readlines()

    with open(os.path.join(data_dir, 'part_%s.txt' %(size)), 'r') as f:
        part = f.readlines()

    with open(os.path.join(data_dir,'%s_aug.txt' %(size)), 'r') as f:
        gesture = f.readlines()

    dir_path = os.path.join(data_dir, 'imglists')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(os.path.join(dir_path,"train_%s_gesture.txt"%(net)), "w") as f:
        nums = [len(neg), len(pos), len(part)]
        ratio = [2, 1, 1] # was 3 1 1 but we think 3 is too much for neg samples
        base_num = min(nums)
        #base_num = 250000
        print('num of neg: ', len(neg))
        print('num of pos: ', len(pos))
        print('num of part: ', len(part))
        print('base num: ', base_num)

        #shuffle the order of the initial data
        #if negative examples are more than 750k then only choose 750k
        if net == 'ONet':

            if len(neg) > base_num * 3: # was 3
                neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True) # was 3
            else:
                neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
            # npr.choice: randomly choose some index to keep
            # i.e. randomly choose from np.arange(len(pos))
            pos_keep = npr.choice(len(pos), size=len(pos), replace=True)
            part_keep = npr.choice(len(part), size=len(part), replace=True)
            print('number of kept neg, pos and part: ')
            print(len(neg_keep), len(pos_keep), len(part_keep))

        else:
            if len(neg) > base_num * 3: # was 3
                neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True) # was 3
            else:
                neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
            # npr.choice: randomly choose some index to keep
            # i.e. randomly choose from np.arange(len(pos))
            pos_keep = npr.choice(len(pos), size=base_num, replace=True)
            part_keep = npr.choice(len(part), size=base_num, replace=True)
            print('number of kept neg, pos and part: ')
            print(len(neg_keep), len(pos_keep), len(part_keep))
        # write the data according to the shuffled order
        for i in pos_keep:
            if not pos[i].startswith('..'):
                pos[i] = '../' + pos[i]
            f.write(pos[i])
        for i in neg_keep:
            if not neg[i].startswith('..'):
                neg[i] = '../' + neg[i]
            f.write(neg[i])
        for i in part_keep:
            if not part[i].startswith('..'):
                part[i] = '../' + part[i]
            f.write(part[i])
        for item in gesture:
            if not item.startswith('..'):
                item = '../' + item
            f.write(item)
