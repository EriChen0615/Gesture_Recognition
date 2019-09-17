import cv2
import os, os.path


def get_lists(path=''):
    with open(os.path.join(path, 'imglist_with_gesture.txt')) as f:
        list_of_img = f.readlines()
    img_list = []
    gt_list = []
    for i in list_of_img:
        i = i.split(" ")
        # print(i)
        img = cv2.imread(os.path.join(path,i[0]))
        img_list.append(img)
        bbox = [int(i[1]), int(i[2]),int(i[3]), int(i[4])]
        # print(bbox)
        print(i[0], bbox)
        gt_list.append(bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    xx1 = np.maximum(box[0], bboxes[:, 0])
    yy1 = np.maximum(box[1], bboxes[:, 1])
    xx2 = np.minimum(box[2], bboxes[:, 2])
    yy2 = np.minimum(box[3], bboxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    over = inter / (box_area + areas - inter)

    return over
    
def main():
    path = 'XW-Dataset/Training'
    img_list, gt_list = get_lists(path)

if __name__ == "__main__":
    main()