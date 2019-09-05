import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from tensorboard.plugins import projector

from mtcnn_config import config

sys.path.append("../Training_Data_Generation")
print(sys.path)
from read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord

import random
import cv2





def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op

'''
certain samples mirror
def random_flip_images(image_batch,label_batch,gesture_batch):
    num_images = image_batch.shape[0]
    random_number = npr.choice([0,1],num_images,replace=True)
    #the index of image needed to flip
    indexes = np.where(random_number>0)[0]
    flipgestureindexes = np.where(label_batch[indexes]==-2)[0]
    
    #random flip    
    for i in indexes:
        cv2.flip(image_batch[i],1,image_batch[i])
    #pay attention: flip gesture    
    for i in flipgestureindexes:
        gesture_ = gesture_batch[i].reshape((-1,2))
        gesture_ = np.asarray([(1-x, y) for (x, y) in gesture_])
        gesture_[[0, 1]] = gesture_[[1, 0]]#left eye<->right eye
        gesture_[[3, 4]] = gesture_[[4, 3]]#left mouth<->right mouth        
        gesture_batch[i] = gesture_.ravel()
    return image_batch,gesture_batch
'''

# all mini-batch mirror
def random_flip_images(image_batch,label_batch, gesture_batch):
    #mirror
    if random.choice([0,1]) > 0:
        #num_images = image_batch.shape[0]
        flipgestureindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((flipgestureindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip gesture   
        #print('gesture batch:',gesture_batch)
         
        # for i in flipgestureindexes:
        #     gesture_ = gesture_batch[i].reshape((-1,2))
        #     gesture_ = np.asarray([(1-x, y) for (x, y) in gesture_])
        #     gesture_[[0, 1]] = gesture_[[1, 0]]#left eye<->right eye
        #     gesture_[[3, 4]] = gesture_[[4, 3]]#left mouth<->right mouth        
        #     gesture_batch[i] = gesture_.ravel()
        
    return image_batch, gesture_batch


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)

    return inputs


def train(net_factory, prefix, end_epoch, base_dir,
          display=100, base_lr=0.01):

    """
    train PNet/RNet/ONet
    :param net_factory: a function defined in mtcnn_model.py
    :param prefix: model path
    :param end_epoch:
    :param dataset:
    :param display:
    :param base_lr:
    :return:

    """
    net = prefix.split('/')[-1]
    #label file
    label_file = os.path.join(base_dir,'train_%s_gesture.txt' % net)
    #label_file = os.path.join(base_dir,'gesture_12_few.txt')
    print(label_file)
    f = open(label_file, 'r')
    # get number of training examples
    num = len(f.readlines())
    print("Total size of the dataset is: ", num)
    print(prefix)

    #PNet use this method to get data
    if net == 'PNet':
        #dataset_dir = os.path.join(base_dir,'train_%s_ALL.tfrecord_shuffle' % net)
        dataset_dir = os.path.join(base_dir,'train_%s_gesture.tfrecord_shuffle' % net)
        print('dataset dir is:',dataset_dir)
        image_batch, label_batch, bbox_batch, gesture_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)
        
    #RNet use 3 tfrecords to get data    
    else:
        pos_dir = os.path.join(base_dir,'pos_gesture.tfrecord_shuffle')
        part_dir = os.path.join(base_dir,'part_gesture.tfrecord_shuffle')
        neg_dir = os.path.join(base_dir,'neg_gesture.tfrecord_shuffle')
        #gesture_dir = os.path.join(base_dir,'gesture_gesture.tfrecord_shuffle')
        gesture_dir = os.path.join('../Dataset/imglists/RNet','gesture_gesture.tfrecord_shuffle')
        dataset_dirs = [pos_dir,part_dir,neg_dir,gesture_dir]
        pos_radio = 1.0/6;part_radio = 1.0/6;gesture_radio=1.0/6;neg_radio=3.0/6
        pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
        assert pos_batch_size != 0,"Batch Size Error "
        part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
        assert part_batch_size != 0,"Batch Size Error "
        neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
        assert neg_batch_size != 0,"Batch Size Error "
        gesture_batch_size = int(np.ceil(config.BATCH_SIZE*gesture_radio))
        assert gesture_batch_size != 0,"Batch Size Error "
        batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,gesture_batch_size]
        #print('batch_size is:', batch_sizes)
        image_batch, label_batch, bbox_batch,gesture_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net)        
        
    #gesture_dir    
    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_gesture_loss = 0.5
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_gesture_loss = 0.5
    else:
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_gesture_loss = 1
        image_size = 48
    
    #define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    gesture_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,3],name='gesture_target')
    #get loss and accuracy
    print(bbox_target)
    print(gesture_target)
    input_image = image_color_distort(input_image)
    cls_loss_op,bbox_loss_op,gesture_loss_op,L2_loss_op,accuracy_op = net_factory(input_image, label, bbox_target,gesture_target,training=True)
    #train,update learning rate(3 loss)
    total_loss_op  = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_gesture_loss*gesture_loss_op + L2_loss_op
    train_op, lr_op = train_model(base_lr,
                                  total_loss_op,
                                  num)
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()

    #save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    tf.summary.scalar("gesture_loss",gesture_loss_op)#gesture_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, gesture loss and L2 loss add together
    summary_op = tf.summary.merge_all()

    time = 'train-{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.now() )
    print("-------------------------------------------------------------\n")
    print("the sub dir's name is: ", time)
    print("-------------------------------------------------------------\n")
    logs_dir = "../logs/%s/" %(net)
    logs_dir = logs_dir + time + "/"
    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer,projector_config)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()

    try:

        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,gesture_batch_array = sess.run([image_batch, label_batch, bbox_batch,gesture_batch])
            #random flip
            image_batch_array,gesture_batch_array = random_flip_images(image_batch_array,label_batch_array,gesture_batch_array)
            '''
            print('im here')
            print(image_batch_array.shape)
            print(label_batch_array.shape)
            print(bbox_batch_array.shape)
            print(gesture_batch_array.shape)
            print(label_batch_array[0])
            print(bbox_batch_array[0])
            print(gesture_batch_array[0])
            '''


            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,gesture_target:gesture_batch_array})

            if (step+1) % display == 0:
                #acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss,gesture_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,gesture_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, gesture_target: gesture_batch_array})

                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_gesture_loss*gesture_loss + L2_loss
                # gesture loss: %4f,
                print("%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,gesture loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                datetime.now(), step+1,MAX_STEP, acc, cls_loss, bbox_loss,gesture_loss, L2_loss,total_loss, lr))


            #save every two epochs
            if i * config.BATCH_SIZE > num*2:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch*2)
                print('path prefix is :', path_prefix)
            writer.add_summary(summary,global_step=step)

    except tf.errors.OutOfRangeError:
        print("Finished!( ゜- ゜)つロ乾杯")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()


def test(net_factory, prefix, end_epoch, base_dir, display=100):

    """
    testing: batch size = 1
    :param net_factory: P/R/ONet
    :param base_dir: tfrecord path
    :param prefix: model path
    :param display:
    :param lr: learning rate
    :return:

    """
    net = prefix.split('/')[-1]
    #label file
    label_file = os.path.join(base_dir,'test_%s_gesture.txt' % net)
    #label_file = os.path.join(base_dir,'gesture_12_few.txt')
    print(label_file)
    f = open(label_file, 'r')
    # get number of testing examples
    num = len(f.readlines())
    print("Total size of the dataset is: ", num)
    print(prefix)

    #PNet use this method to get data
    #if net == 'PNet':
        #dataset_dir = os.path.join(base_dir,'train_%s_ALL.tfrecord_shuffle' % net)
    dataset_dir = os.path.join(base_dir,'test_%s_gesture.tfrecord_shuffle' % net)
    print('dataset dir is:',dataset_dir)
    image_batch, label_batch, bbox_batch, gesture_batch = read_single_tfrecord(dataset_dir, 1, net)
    image_size = 12
    radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_gesture_loss = 0.5
        
    # else 之后再写吧lol：need to use multi_tfrecord reader
    """ for RNET & ONET """
    #else:
    
    

    #set placeholders first 
    #change batchsize to 1 for testing
    input_image = tf.placeholder(tf.float32, shape=[1, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[1], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[1, 4], name='bbox_target')
    gesture_target = tf.placeholder(tf.float32,shape=[1,3],name='gesture_target')

    input_image = image_color_distort(input_image)
    
    cls_loss_op,bbox_loss_op,gesture_loss_op,L2_loss_op,accuracy_op = net_factory(input_image, label, bbox_target,gesture_target,training=False)
    #train,update learning rate(3 loss)
    total_loss_op  = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_gesture_loss*gesture_loss_op + L2_loss_op
    base_lr = 0
    train_op, lr_op = train_model(base_lr,
                                  total_loss_op,
                                  num) #for testing, set base lr to 0
    

    #cls_pro_test,bbox_pred_test,gesture_pred_test = net_factory(input_image, label, bbox_target,gesture_target,training=False)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()

    #save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)
    tf.summary.scalar("bbox_loss",bbox_loss_op)
    tf.summary.scalar("gesture_loss",gesture_loss_op)
    
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs_testing/%s" %(net)
    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer,projector_config)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()
    
    try:

        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,gesture_batch_array = sess.run([image_batch, label_batch, bbox_batch,gesture_batch])
            #random flip
            image_batch_array,gesture_batch_array = random_flip_images(image_batch_array,label_batch_array,gesture_batch_array)

            _,_,summary = sess.run([train_op, lr_op, summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,gesture_target:gesture_batch_array})

            if (step+1) % display == 0:
                
                cls_loss,bbox_loss,gesture_loss, accuracy = sess.run([cls_loss_op,bbox_loss_op,gesture_loss_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, gesture_target: gesture_batch_array})

                #total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_gesture_loss*gesture_loss + L2_loss
                # gesture loss: %4f,
                print("%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f, gesture_loss: %4f  " % (datetime.now(), step+1,MAX_STEP, accuracy, cls_loss,bbox_loss,gesture_loss))


            #save every two epochs
            if i * config.BATCH_SIZE > num*2:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch*2)
                print('path prefix is :', path_prefix)
            
            writer.add_summary(summary,global_step=step)


    except tf.errors.OutOfRangeError:
        print("Finished!( ゜- ゜)つロ乾杯")
    finally:
        coord.request_stop()
        writer.close()

    coord.join(threads)
    sess.close()

    