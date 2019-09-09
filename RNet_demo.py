# coding:utf-8
import sys
import fnmatch

sys.path.append('..')
# from Detector.MtcnnDetector import MtcnnDetector
from Train_Model.mtcnn_config import config
from Detector.RnetDetector import RnetDetector
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

test_mode = "RNet"
thresh = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
model_path = ['Model/PNet/PNet-500', 'Model/RNet/RNet_No_Landmark/RNet-500', '']
epoch = [500, 500, 16]
batch_size = [300, 300, 300]
print(model_path)

TestImage_path = "Testing_Demo_Data/webimg/"
TestResult_path = "RNet_demo/ResultImage/webimg/"

mkdir(TestResult_path)

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

#  ---------- R&O Net Model -------------------
# # load onet model
# if test_mode == "ONet":
#     ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
#     detectors[2] = ONet

mtcnn_detector = RnetDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
gt_imdb = []

# gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
# imdb_ = dict()"
# imdb_['image'] = im_path
# imdb_['label'] = 5


_, img_list = scan_file(TestImage_path, 'jpg')
for item in img_list:
    gt_imdb.append(os.path.join(TestImage_path, item))
test_data = TestLoader(gt_imdb)

print(test_data)
all_boxes, landmarks = mtcnn_detector.detect_face(test_data)
print(len(landmarks))

count = 0

for imagepath in gt_imdb:
    print(imagepath)
    image = cv2.imread(imagepath)
    # image_original = image.copy()
    for box_number, bbox in enumerate(all_boxes[count]):
        cv2.putText(image, str(np.round(bbox[4], 2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    color=(255, 0, 255))
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))
        # image_single = image_original.copy()
        # cv2.rectangle(image_single, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))
        # cv2.imwrite("{}/{}_{}.png".format(TestResult_path, count, box_number), image_single)
        # with open("{}/{}_{}.txt".format(TestResult_path, count, box_number), 'w') as f:
        #     f.write('(x1,y1):({},{})\n(x2,y2):({},{})\nprediction:{}'.format(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), np.round(bbox[4], 4)))



        # for landmark in landmarks[count]:
        # for i in range(len(landmark)//2):
        #     cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))


    count = count + 1
    # cv2.imwrite("result_landmark/%d.png" %(count),image)
    cv2.imwrite("{}/{}.png".format(TestResult_path, count-1), image)
    # cv2.imshow("PNet", image)
    # cv2.waitKey(0)

'''
for data in test_data:
    print type(data)
    for bbox in all_boxes[0]:
        print bbox
        print (int(bbox[0]),int(bbox[1]))
        cv2.rectangle(data, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
    #print data
    cv2.imshow("lala",data)
    cv2.waitKey(0)
'''