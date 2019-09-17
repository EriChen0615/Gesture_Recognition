# coding:utf-8
import sys
import fnmatch

sys.path.append('..')
from Train_Model.mtcnn_config import config
from Detector.MtcnnDetector import MtcnnDetector
from Detector.detector import Detector
from Detector.fcn_detector import FcnDetector
from Train_Model.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import os, os.path
import numpy as np

def scan_file(file_dir = '', file_postfix = 'jpg'):
    '''
    This function will scan the file in the given directory and return the number
    and file name list for files satisfying the postfix condition.
    :param file_dir: string, should end with '/';
    :param file_type: string, no need for '.';
    :return: file_count: list of file names whose postfix satisfies the condition;
    '''
    file_count = 0
    file_list = []
    for f_name in os.listdir(file_dir):
        if fnmatch.fnmatch(f_name, ('*.' + file_postfix)):
            file_count += 1
            file_list.append(f_name)
    return file_count, file_list

class TestLoader:
    # imdb image_path(list)
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)  # num of data
        # self.index = np.arange(self.size)

        self.cur = 0
        self.data = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            # shuffle test image
            np.random.shuffle(self.imdb)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    # realize __iter__() and next()--->iterator
    # return iter object
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        imdb = self.imdb[self.cur]
        '''
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        #picked image
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        # print(imdb)
        '''
        # print type(imdb)
        # print len(imdb)
        # assert len(imdb) == 1, "Single batch only"
        im = cv2.imread(imdb)
        # print("=======test loader======")
        # print(im.shape)
        # cv2.imshow('im', im)
        # cv2.waitKey(0)
        self.data = im

class ImageLoader:
    def __init__(self, imdb, im_size, batch_size=config.BATCH_SIZE, shuffle=False):

        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names = ['label', 'bbox_target']
        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data, self.label
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = minibatch.get_minibatch(imdb, self.num_classes, self.im_size)
        self.data = data['data']
        self.label = [label[name] for name in self.label_names]

def mkdir(path):
    path = path.strip()
    path= path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' created successfully')
        return True
    else:
        print(path + ' already exist')
        return False

def evaluate(box, gt, scale_factor=1):

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)

    # inter area
    xx1 = np.maximum(box[0], gt[0])
    yy1 = np.maximum(box[1], gt[1])
    xx2 = np.minimum(box[2], gt[2])
    yy2 = np.minimum(box[3], gt[3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h

    # Centre of predicted bounding box
    cbx = (box[0] + box[2])/ 2
    cby = (box[1] + box[3])/ 2

    # Centre of ground truth bounding box
    cgx = (gt[0] + gt[2])/ 2
    cgy = (gt[1] + gt[3]) / 2

    # Deviation of centre
    dx = abs(cbx - cgx)
    dy = abs(cby - cgy)
    deviation_square = (dx**2 + dy**2)

    exp_index = deviation_square / box_area

    PNet_index = (inter / gt_area) * (inter / box_area) * scale_factor / (np.exp(exp_index))
    return PNet_index

def get_lists(path=''):
    with open(os.path.join(path, 'imglist_with_gesture.txt')) as f:
        list_of_img = f.readlines()
    img_list = []
    gt_list = []
    for i in list_of_img:
        i = i.split(" ")
        # print(i)
        img_list.append(os.path.join(path, i[0]))
        bbox = [int(i[1]), int(i[2]), int(i[3]), int(i[4])]
        # print(bbox)
        # print(bbox)
        gt_list.append(bbox)
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return img_list, gt_list

def IoU(box, bboxes):
    """
    Caculate IoU between detect and ground truth boxes
    :param crop_box:numpy array (4, )
    :param bboxes:numpy array (n, 4):x1, y1, x2, y2
    :return:
    numpy array, shape (n, ) Iou
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas = (bboxes[2] - bboxes[0] + 1) * (bboxes[3] - bboxes[1] + 1)
    xx1 = np.maximum(box[0], bboxes[0])
    yy1 = np.maximum(box[1], bboxes[1])
    xx2 = np.minimum(box[2], bboxes[2])
    yy2 = np.minimum(box[3], bboxes[3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    over = inter / (box_area + areas - inter)

    return over

def main(test_mode="ONet"):

    # Configuration mode
    print("===============================")
    print("> Test mode is {}".format(test_mode))
    print("===============================")

    # Initialise
    slide_window = False
    detectors = [None, None, None]
    thresh = [0.6, 0.7, 0.7]
    IoU_upper_threshold = 0.65
    IoU_lower_threshold = 0.3
    min_face_size = 20
    stride = 2
    batch_size = [2048, 64, 16]

    # The model path, should be the same in the checkpoint file
    model_path = ['Model/PNet/PNet-30', 'Model/RNet/RNet-500', 'Model/ONet/ONet-116']
    # Test image path
    path = 'XW-Dataset/Training'


    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet


    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet


    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    # Initialise detcector
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    # Get the test img list
    img_list, gt_list = get_lists(path)
    print('img number: ', len(img_list), len(gt_list))

    test_data = TestLoader(img_list)

    # Get boxes and landmarks
    all_boxes, landmarks = mtcnn_detector.detect_face(test_data)

    FP_set  = [] # False Positive (False Alarm - IoU < IoU_lower_threshold);
    FN_img  = [] # False Negative (Detection Failure - if there is no bbox whose IoU > IoU_upper_threshold);
    TP_set  = [] # True  Positive (Correct Detection - IoU > IoU_upper_threshold);
    NM_set  = [] # Normal bounding boxes (IoU_lower_threshold < IoU < IoU_upper_threshold);
    FPR_set = [] # False Positive Rate (FP / total boxes);
    TPR_set = [] # False Positive Rate (TP / total boxes);
    score_set = [] # Scores

    for count, gt in enumerate(gt_list):
        TP_img = [] # True  Positive bbox for each img;
        FP_img = [] # False Positive bbox for each img;
        NM_img = [] # Normal bbox for each img.
        score_img = []

        # Evaluate scores per picture

        # Draw boxes
        for box_number, bbox in enumerate(all_boxes[count]):
            # Evaluate box
            iou = IoU(bbox[:-1], gt)
            score = evaluate(bbox[:-1], gt)
            score_img.append(score)
            # print(box_number, iou)
            if iou > IoU_upper_threshold:
                TP_img.append(bbox)
            elif iou < IoU_lower_threshold:
                FP_img.append(bbox)
            else:
                NM_img.append(bbox)

            # Skip the landmarks if it is PNet mode
            if test_mode == "PNet":
                continue

        if len(all_boxes[count])!=0:
            FPR = len(FP_img)/len(all_boxes[count])
            FPR_set.append(FPR)
            TPR = len(TP_img)/len(all_boxes[count])
            TPR_set.append(TPR)
        TP_set.append(TP_img)
        FP_set.append(FP_img)
        NM_set.append(NM_img)
        score_set.append(sum(score_img)/len(score_img))
        if not TP_img:
            FN_img.append(count)



    # print result
    print('=================== Result =====================\n')

    print(" Testing samples: {};\n Testing Model: {};\n Testing Net: {}.\n".format(len(img_list), model_path, test_mode))

    print("1. Max FPR: {}, Min FPR: {}, Average FPR: {}".
          format(round(max(FPR_set), 4), round(min(FPR_set), 4), round(sum(FPR_set) / len(FPR_set), 4)))
    print("   FPR(False Positive Rate) refers to the ratio of # mistakenly marked\n   bounding boxes and # total boxes for each image.\n   IoU < {}\n".format(IoU_lower_threshold))

    print("2. Max TPR: {}, Min TPR: {}, Average TPR: {}".
          format(round(max(TPR_set), 4), round(min(TPR_set), 4), round(sum(TPR_set) / len(TPR_set), 4)))
    print("   TPR(True Positive Rate) refers to the ratio of # successfully marked\n   bounding boxes and # total boxes for each image.\n   IoU > {}\n".format(IoU_upper_threshold))

    print("3. Max score: {}, Min score: {}, Average score: {}".
          format(round(max(score_set),4), round(min(score_set),4), round(sum(score_set)/len(score_set),4)))
    print(
        "   Scores are calculated following sum(score*IoU)/sum(score) for each image.\n")

    # print(FN_img)
    print("4. False Negative Rate: {}".format(round(len(FN_img)/len(img_list), 4)))
    print("   False Negative Rate refers to the ratio of # images which never have\n   positive boxes and # total images for the whole testing set.\n")

    print("Images whose indices are in the following list do not have positive bounding boxes:")
    print(FN_img)

if __name__ == '__main__':
    test_mode = sys.argv[1]
    assert test_mode == "PNet" or test_mode == "RNet" or test_mode == "ONet", "Invalid test mode, please check your input."
    path = 'Dataset/Training'
    img_list, gt_list = get_lists(path)
    main(test_mode)