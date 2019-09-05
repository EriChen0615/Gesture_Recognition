"""
This file is used to generate tf_records for training of PNet

"""

import os
import random
import sys
import time

import tensorflow as tf

from tf_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      filename: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    #print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    image_data, _, _ = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    return '%s/train_PNet_gesture.tfrecord' % (output_dir)
    

def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, net=net)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        #random.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord

    print('lalala')

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i+1) % 100 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted' % (i+1, len(dataset)))
                #sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            if filename[0]!='.':
                filename = os.path.join('..',filename)
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir, net='PNet'):
    #get file name , label and anotation
    #item = 'imglists/PNet/train_%s_raw.txt' % net
    item = 'imglists/PNet/train_%s_gesture.txt' % net
    
    dataset_dir = os.path.join(dir, item)

    print(dataset_dir)
    imagelist = open(dataset_dir, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict() # neg(0) & aug(-2)
        data_example['filename'] = info[0]
        #print(data_example['filename'])
        data_example['label'] = int(info[1])

        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['gesture_one'] = 0
        bbox['gesture_fist'] = 0
        bbox['gesture_two'] = 0


        if len(info)==6: # pos(1) & part(-1)

    
            bbox['xmin'] = info[2]
            bbox['ymin'] = info[3]
            bbox['xmax'] = info[4]
            bbox['ymax'] = info[5]

        if len(info)==5: #aug

            bbox['gesture_one'] = info[2]
            bbox['gesture_fist'] = info[3]
            bbox['gesture_two'] = info[4]




        """
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 9:
            bbox['gesture_one'] = float(info[6])
            bbox['gesture_two'] = float(info[7])
            bbox['gesture_three'] = float(info[8])

        """

        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    dir = '../Dataset/Training/'
    net = 'RNet'
    output_directory = '../Dataset/Training/imglists/RNet'
    run(dir, net, output_directory, shuffling=True)
