import os, os.path
import fnmatch
import numpy as np
import random 
import shutil

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

def read_txt(path, file):
    with open(os.path.join(path,file), 'r') as f:
        imglist = f.read().splitlines()
        print('There are {} lines in the txt'.format(len(imglist)))
        for i in range(len(imglist)):
            imglist[i] = imglist[i].replace('    ', ' ')
    return imglist

def write_config(file, base_dir, groups, count_list, count):
    f = open(os.path.join(base_dir, file), 'w')
    f.write('@ {}\n'.format(str(count)))
    for i in range(len(groups)):
        # print(i)
        f.write('! {} {}\n'.format(count_list[i], groups[i]))
    for i in groups:
        # print(i)
        f.write('- {} {} \n'.format(os.path.join(base_dir,i), i) )
    f.close()
    

def main():
    train_dir = '../ego_data/Training/'
    mkdir(train_dir)
    test_dir = '../ego_data/Testing/'
    mkdir(test_dir)
    label_dir = '../ego_data/label/'
    img_base_dir = '../ego_data/JPG/'
    img_groups = np.array(['SingleBad', 'SingleGood', 'SingleOne', 'SingleTwo', 'SingleFour',
             'SingleSix', 'SingleEight', 'SingleNine'])
    label_names = [img_group + '.txt' for img_group in img_groups]
    # print(label_names)
    test_num = 50
    random.seed(0)

    train_count = 0
    train = []
    test_count = 0
    test = []

    for i, img_group in enumerate(img_groups):
        total_list = read_txt(label_dir, label_names[i])
        train_dest = os.path.join(train_dir, img_group)
        mkdir(train_dest)
        test_dest = os.path.join(test_dir, img_group)
        mkdir(test_dest)
        img_dir = os.path.join(img_base_dir,img_group)
        img_num, _ = scan_file(img_dir)
        print('-----------------------------------------------')
        print('{}: there are {} images in total'.format(img_group, img_num))
        print('there are {} labels in total'.format(len(total_list)))
        test_inds = random.sample(list(np.arange(len(total_list))), test_num)
        print(len(total_list))
        print('test_inds chosen: ', test_inds)
        test_list = [total_list[i] for i in test_inds]
        print('{} testing samples chosen for testing'.format(len(test_list)))
        test_count += len(test_list)
        test.append(len(test_list))
        train_list = np.delete(total_list, test_inds)
        print('{} images left in the training dataset'.format(len(train_list)))
        train_count += len(train_list)
        train.append(len(train_list))
        print('-----------------------------------------------\n')

        train_writer = open(os.path.join(train_dest, 'annotation.txt'), 'w')
        test_writer = open(os.path.join(test_dest, 'annotation.txt'), 'w')
        

        for test_img in test_list:
            # print(test_img)
            # print(type(test_img))
            test_writer.write(test_img + '\n')
            test_img = test_img.strip().split(" ")
            shutil.copy(os.path.join(img_base_dir, test_img[0]), os.path.join(test_dir, test_img[0]))
        for train_img in train_list:
            train_writer.write(train_img + '\n')
            train_img = train_img.strip().split(" ")
            shutil.copy(os.path.join(img_base_dir, train_img[0]), os.path.join(train_dir, train_img[0]))
        print('writing annotation files completed')
        print('-----------------------------------------------\n')
        train_writer.close()
        test_writer.close()
    
    print('-----------------------------------------------')
    print('total number of testing samples: ', test_count)
    print(test)
    print('total number of training samples: ', train_count)
    print(train)
    print('-----------------------------------------------\n')
    write_config('config.txt', test_dir, img_groups, test, test_count)
    write_config('config.txt', train_dir, img_groups, train, train_count)
    print('config files writing finished')


if __name__ == "__main__":
    main()




    


