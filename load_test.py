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
        print(bbox)
        gt_list.append(bbox)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_list, gt_list

def main():
    path = 'Dataset/Training'
    img_list, gt_list = get_lists(path)

if __name__ == "__main__":
    main()