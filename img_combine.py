from PIL import Image
import os
import fnmatch
import sys

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

def main(mode):
    file_path_1 = "MTCNN_demo/PNet/ResultImage/time/"
    file_path_2 = "MTCNN_demo/RNet/ResultImage/time/"
    file_path_3 = "MTCNN_demo/ONet/ResultImage/time/"
    dst_path = "MTCNN_demo/time/"
    # print("=====please type in file path=====")
    # file_path = []
    # for i in range(mode):
    #     print("suggested path:" + "MTCNN_demo/PNet/ResultImage/XW_Dataset/")
    #     file_path.append = input("File path:")
    # print("suggested dst_path: MTCNN_demo/Sep13_PRO/")
    # dst_path = input("dst path:")

    mkdir(dst_path)

    file_count, file_list = scan_file(file_path_1, 'png')

    for img in file_list:

        img_path_1 = file_path_1 + img
        img_path_2 = file_path_2 + img
        if mode == "3":
            img_path_3 = file_path_3 + img

        im1 = Image.open(img_path_1)
        im2 = Image.open(img_path_2)
        if mode == "3":
            im3 = Image.open(img_path_3)

        width, height = im1.size

        if mode == "3":
            result = Image.new(im1.mode, (width * 3, height))
            result.paste(im3, box=(width * 2, 0))
        else:
            result = Image.new(im1.mode, (width * 2, height))
        result.paste(im2, box=(width, 0))
        result.paste(im1, box=(0, 0))

        result.save(dst_path+img)


if __name__ == '__main__':
    combine = sys.argv[1]
    assert combine == "2" or combine == "3", "Invalid input, please check your input."
    main(combine)