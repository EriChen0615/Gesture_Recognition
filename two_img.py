from PIL import Image
import os
import fnmatch

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

def main():
    file_path_1 = "MTCNN_demo/PNet-Sep/ResultImage/Test/"
    file_path_2 = "MTCNN_demo/PNet/ResultImage/Test/"
    dst_path = "MTCNN_demo/Test_Two_img/"

    mkdir(dst_path)

    file_count, file_list = scan_file(file_path_1, 'png')

    for img in file_list:
        img_path_1 = file_path_1 + img
        img_path_2 = file_path_2 + img

        im1 = Image.open(img_path_1)
        im2 = Image.open(img_path_2)

        width, height = 240, 240

        result = Image.new(im1.mode, (width * 2, height))
        result.paste(im2, box=(width, 0))
        result.paste(im1, box=(0, 0))

        result.save(dst_path+img)


if __name__ == '__main__':
    main()