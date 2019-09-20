#coding:utf-8
import sys
#sys.path.append("../")
from utils import convert_to_square

sys.path.insert(0,'..')
import numpy as np
import argparse
import os
import pickle as pickle
import cv2
from Train_Model.mtcnn_model import P_Net, R_Net, O_Net
from Train_Model.mtcnn_config import config
from loader import TestLoader
from Detector.detector import Detector
from Detector.fcn_detector import FcnDetector
from Detector.MtcnnDetector import MtcnnDetector
from utils import *
from data_utils import *
import numpy.random as npr
import os.path

#net : 24(RNet)/48(ONet)
#data: dict()

def save_hard_example(net, data,save_path):
    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    im_idx_list = data['images']
    # print(images[0])
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    print("processing %d images in total" % num_of_images)

    
    # save files
    neg_label_file = "%s/neg_%d.txt" % (save_path, image_size)
    neg_file = open(neg_label_file, 'w')

    pos_label_file = "%s/pos_%d.txt" % (save_path, image_size)
    pos_file = open(pos_label_file, 'w')

    part_label_file = "%s/part_%d.txt" % (save_path, image_size)
    part_file = open(part_label_file, 'w')
    #read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    # print(len(det_boxes), num_of_images)
    print(len(det_boxes))
    print(num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    #im_idx_list image index(list)
    #det_boxes detect result(list)
    #gt_boxes_list gt(list)
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue

        img = cv2.imread(im_idx)
        
        #change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3            
            if np.max(Iou) < 0.3 and neg_num < 60:
                #save the examples
                save_file = get_path(neg_dir, "%s.jpg" % n_idx)
                # print(save_file)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

                if image_size==48: # augment ONet negative samples by 10 times
                    while neg_num < 10:
                        #neg_num's size [40,min(width, height) / 2],min_size:40
                        # size is a random number between 12 and min(width,height)
                        size = npr.randint(-min(width,height)/5, min(width, height) /5)
                        #top_left coordinate
                        #nx = npr.randint(0, width - size)
                        #ny = npr.randint(0, height - size)
                        #random crop
                        crop_box = np.array([max(x_left+size,0), max(0,y_top+size), x_right+size, y_bottom+size])
                        #calculate iou
                        Iou = IoU(crop_box, gts)
                        print(neg_num)
                        #crop a part from inital image
                        cropped_im = img[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2], :]
                        #resize the cropped image to size 12*12
                        resized_im = cv2.resize(cropped_im, (48, 48), interpolation=cv2.INTER_LINEAR)

                    if np.max(Iou) < 0.3:
                        # Iou with all gts must below 0.3
                        save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                        neg_file.write(save_file + ' 0\n')
                        cv2.imwrite(save_file, resized_im)
                        n_idx += 1
                        neg_num += 1

            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = get_path(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


def t_net(prefix, epoch,
             batch_size, test_mode="PNet",
             thresh=[0.6, 0.6, 0.7], min_face_size=25,
             stride=2, slide_window=False, shuffle=False, vis=False,im_dir=None,filename='imglist_with_gesture.txt'):
    detectors = [None, None, None]
    print("Test model: ", test_mode)
    #PNet-echo
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)] # path model include epoch number
    print(model_path[0]) # load the first model only
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        print("==================================", test_mode)
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        print("==================================", test_mode)
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet
        
    basedir = im_dir
    #anno_file
    filename = 'imglist_with_gesture.txt'
    #read anotation(type:dict), include 'images' and 'bboxes'
    #data = read_annotation(basedir,filename)
    data = load_annotation(os.path.join(basedir,filename)) # modified version of load annotation
    # print('to see if load_annotation works fine: ')
    # print('data:',data)
    '''
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    #print("==================================")
    # 注意是在“test”模式下
    # imdb = IMDB("wider", image_set, root_path, dataset_path, 'test')
    # gt_imdb = imdb.gt_imdb()
    print('load test data')
    test_data = TestLoader(data['images'])
    print('test data example: ', test_data.imdb[0])
    print ('finish loading')
    #list
    print ('start detecting....')
    detections,_ = mtcnn_detector.detect_face(test_data)
    print('detections:',detections)
    print ('finish detecting ')
    save_net = 'RNet'
    if test_mode == "PNet":
        save_net = "RNet"
    elif test_mode == "RNet":
        save_net = "ONet"
    #save detect result
    '''
    save_path = data_dir
    print ('save_path is :')
    print(save_path)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # save_file = os.path.join(save_path, "detections.pkl")
    # with open(save_file, 'wb') as f:
    #     pickle.dump(detections, f,1)
    print("%s testing finished, start OHEM" % image_size)
    save_hard_example(image_size, data, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--im_dir',dest='im_dir',help='base directory for the dataset')
    parser.add_argument('--anno_file',dest='anno_file',help='path under im_dir of the annotation file')
    parser.add_argument('--save_dir',dest='save_dir',help='the directory to save the gesture data')
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='RNet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['../Model/MTCNN/PNet','../Model/MTCNN/RNet_No_Landmark/RNet'],
                        type=str) # model file location
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[500,300], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.6, 0.7, 0.8], type=float) # note default value for PNet is very low
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=20, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    #parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    
    net = args.test_mode

    if net == "PNet":
        image_size = 24
    if net == "RNet":
        image_size = 48

    base_dir = args.im_dir
    data_dir = args.save_dir
    
    neg_dir = get_path(data_dir, 'negative')
    pos_dir = get_path(data_dir, 'positive')
    part_dir = get_path(data_dir, 'part')
    #create dictionary shuffle   
    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    print('Called with argument:')
    print(args)
    t_net(args.prefix,#model param's file
          args.epoch, #final epoches
          args.batch_size, #test batch_size 
          args.test_mode,#test which model
          args.thresh, #cls threshold
          args.min_face, #min_face
          args.stride,#stride
          args.slide_window, 
          args.shuffle, 
          vis=False,
          im_dir=base_dir,
          filename=args.anno_file
          )
