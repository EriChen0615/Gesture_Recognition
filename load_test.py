import cv2
import os, os.path


def get_lists(img_dir='', label_dir='', label_name=''):
    with open(os.path.join(label_dir, label_name)) as f:
        list_of_img = f.readlines()
    img_list = []
    gt_list = []
    x = 0
    for i in list_of_img:
        i = i.split("    ") # need 4 spaces here
        print(i)
        print(len(i))
        img = cv2.imread(os.path.join(img_dir,i[0]))
        # print(open(os.path.join(img_dir,i[0])))
        print(img.size)
        img_list.append(img)
        print([i[1], i[2], i[3], i[4]])
        bbox = [float(i[1])*640, float(i[2])*480,float(i[3])*640, float(i[4])*480]
        # print(bbox)
        print(bbox)
        gt_list.append(bbox)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if x==5:
            break
        x += 1
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
    label_name = 'SingleBad.txt'
    label_dir = 'ego_data/label'
    img_dir = 'ego_data/JPG'
    img_list, gt_list = get_lists(img_dir, label_dir, label_name)
    print(img_list)
    print(gt_list)


if __name__ == "__main__":
    main()