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
import os
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

def demo(test_mode="ONet"):

    # Configuration mode
    print("===============================")
    print("> Test mode is {}".format(test_mode))
    print("===============================")

    # Initialise
    slide_window = False
    detectors = [None, None, None]
    thresh = [0.6, 0.7, 0.7]
    min_face_size = 20
    stride = 2
    batch_size = [2048, 64, 16]

    # The model path, should be the same in the checkpoint file
    model_path = ['Model/PNet/PNet-30', 'Model/RNet/RNet-500', 'Model/ONet/ONet-116']
    # The sub-folder in the folder Testing_Demo_Data
    TestImage_subfolder = "test"
    # Test image postfix
    Image_postfix = 'jpg'

    TestImage_path = "Testing_Demo_Data/{}/".format(TestImage_subfolder)
    TestResult_path = "MTCNN_demo/{}/ResultImage/{}/".format(test_mode, TestImage_subfolder)
    mkdir(TestResult_path)
    if test_mode in ["RNet", "ONet"]:
        mkdir(TestResult_path+'prediction/')


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
    gt_imdb = []

    # Get the test img list
    _, img_list = scan_file(TestImage_path, Image_postfix)
    for item in img_list:
        gt_imdb.append(os.path.join(TestImage_path, item))
    test_data = TestLoader(gt_imdb)

    # Get boxes and landmarks
    all_boxes, landmarks = mtcnn_detector.detect_face(test_data)

    count = 0

    for image_path in gt_imdb:
        print(image_path)
        image = cv2.imread(image_path)
        image_original = image.copy()
        for box_number, bbox in enumerate(all_boxes[count]):
            # Process img with all boxes
            cv2.putText(image, str(np.round(bbox[4], 2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        color=(255, 0, 255))
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))

            # Draw Single Box img
            image_single = image_original.copy()
            cv2.rectangle(image_single, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))


            # Skip the landmarks if it is PNet mode
            if test_mode == "PNet":
                continue

            class_list = np.array(landmarks[count][box_number])
            pred_index = np.argmax(class_list)
            pred_text = str(pred_index)
            if pred_index==0:
                pred_text = "One"
            elif pred_index==1:
                pred_text = 'Fist'
            elif pred_index==2:
                pred_text = 'Two'
            else:
                # if failed to classify, a txt file will be generated
                with open("{}/prediction/{}_{}.txt".format(TestResult_path, count, box_number), 'w') as f:
                    f.write('prediction list:{}'.format(class_list))
                print('file saved!')

            # Save img with single boxes in sub-folder prediction
            cv2.putText(image_single, pred_text, (120,120), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(0, 255, 0))

            cv2.imwrite("{}/prediction/{}_{}.png".format(TestResult_path, count, box_number), image_single)


        count = count + 1

        # Save img with all boxes.
        cv2.imwrite("{}/{}.png".format(TestResult_path, count-1), image)

if __name__ == '__main__':
    test_mode = sys.argv[1]
    assert test_mode == "PNet" or test_mode == "RNet" or test_mode == "ONet", "Invalid test mode, please check your input."
    demo(test_mode)