import numpy as np
import numpy.random as npr
import os

data_dir = '../Dataset/Testing'
#anno_file = os.path.join(data_dir, "annotation.txt")

size = 12

if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"

with open(os.path.join(data_dir, 'pos_%s.txt' %(size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir, 'neg_%s.txt' %(size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, 'part_%s.txt' %(size)), 'r') as f:
    part = f.readlines()

with open(os.path.join(data_dir,'gesture_%s_aug.txt' %(size)), 'r') as f:
    gesture = f.readlines()

dir_path = os.path.join(data_dir, 'imglists')

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
    os.makedirs(os.path.join(dir_path, "%s" %(net)))

with open(os.path.join(dir_path, "%s" %(net),"test_%s_gesture.txt" % (net)), "w") as f:
    nums = [len(neg), len(pos), len(part)]
    ratio = [3, 1, 1]
    base_num = min(nums)
    #base_num = 250000
    print(len(neg), len(pos), len(part), base_num)

    #shuffle the order of the initial data
    #if negative examples are more than 750k then only choose 750k
    #npr.choice: generate a random sample from a given 1D array
    if len(neg) > base_num * 3:
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
    else:
        neg_keep = npr.choice(len(neg), size=len(neg), replace=True)

    pos_keep = npr.choice(len(pos), size=base_num, replace=True)
    part_keep = npr.choice(len(part), size=base_num, replace=True)
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
