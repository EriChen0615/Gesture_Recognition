import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

num_keep_radio = 0.7 # ratio for online hard sample mining

#define prelu: an activation function
def prelu(inputs):
    #set a tensor alphas the same shape as the last dimension of inputs, and initialize its elements to be all 0.25
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs) #pos has the same dimension as inputs, max(inputs,0) 将输入小�的值赋值为0，输入大�的值不�
    neg = alphas * (inputs-abs(inputs))*0.5 
    return pos + neg

def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

#online hard example mining
def cls_ohem(cls_prob, label, training=True):
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0 net_factory

    #pos -> 1, neg -> 0, others -> 0
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    """
    The condition tensor acts as a mask that chooses, based on the value at each element, 
    whether the corresponding element / row in the output should be taken from x (if true)
    or y (if false).
    
    tf.less() Returns the truth value of (x < y) element-wise
    """
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1]) # <tensor>,<shape> flattened
    label_int = tf.cast(label_filter_invalid,tf.int32)
    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    #row = [0,2,4.....]

    row = tf.range(num_row)*2 # [0,2,4..,(num_row-1)*2], because cls_prob was (-1,2), with the even items representing postive detection
    indices_ = row + label_int # valid label of either 0 or 1
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_)) # if img is neg, choose the odd one and drive it to 1; if pos drive even to 1
    loss = -tf.log(label_prob+1e-10) # if 1 loss close to 0; otherwise 10 <driving the label_prob to 1>
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)

    # set pos and neg to be 1, rest to be 0
    # print(label.get_shape())
    # print(zeros.get_shape())
    # print(ones.get_shape())
    valid_inds = tf.where(label < zeros,zeros,ones) #was < before 
    # the coordinates of 'True' elements of the condition given
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)


    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32) # discard some loss
    #FILTER OUT PART AND gesture DATA

    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num) # Finds values and indices of the k largest entries for the last dimension. _ is indices
    return tf.reduce_mean(loss) # take the mean of loss

def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)

def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

#label=1 or label=-1 then do regression !!!this is not doing ohem!!!
def bbox_ohem(bbox_pred,bbox_target,label):
    '''

    :param bbox_pred:
    :param bbox_target:
    :param label: class label
    :return: mean euclidean loss for all the pos and part examples
    '''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    # keep pos and part examples
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    #(batch,)
    #calculate square sum
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    # count the number of pos and part examples
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    # keep top k examples, k equals to the number of positive examples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)

def gesture_ohem(gesture_pred,gesture_target,label): 
    '''

    :param gesture_pred:
    :param gesture_target:
    :param label:
    :return: mean euclidean loss
    '''
    #keep label =-2  then do gesture detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(gesture_pred-gesture_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):
    '''

    :param cls_prob:
    :param label:
    :return:calculate classification accuracy for pos and neg examples only
    '''
    # 0 for negative 1 for positive
    # print("shape of the cls_prob: ")
    # print(cls_prob.get_shape())
    # print("shape of the label: ")
    # print(label.get_shape())
    pred = tf.argmax(cls_prob,axis=1) # get the index of max value along axis1 of cls_prob
    label_int = tf.cast(label,tf.int64) # convert element in label to type int
    cond = tf.where(tf.greater_equal(label_int,0))# return the index of pos and neg examples
    picked = tf.squeeze(cond)
    # gather the label of pos and neg examples
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    #calculate the mean value of a vector contains 1 and 0, 1 for correct classification, 0 for incorrect
    # ACC = (TP+FP)/total population
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op

def _activation_summary(x):

    '''
    creates a summary provides histogram of activations
    creates a summary that measures the sparsity of activations

    :param x: Tensor
    :return:
    '''

    tensor_name = x.op.name
    print('load summary for : ',tensor_name)
    tf.summary.histogram(tensor_name + '/activations',x)
    #tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

#construct Pnet
#label:batch
def P_Net(inputs,label=None,bbox_target=None,gesture_target=None,training=False):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):

        print(inputs.get_shape())


        net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1') # <input>,<number of output>,<kernel size>
                            # 3 interpreted as [3,3]
        _activation_summary(net)
        print(net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        _activation_summary(net)
        print(net.get_shape())
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        _activation_summary(net)
        print(net.get_shape())
        
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        _activation_summary(net)
        print(net.get_shape())

        """ hand detection """
        #batch*H*W*2 shape=(batch,1,1,2) 
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        _activation_summary(conv4_1)
        #conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)
        print ('conv4_1.shape=',conv4_1.get_shape())

        """ bbox regression """
        #batch*H*W*4 shape=(batch,1,1,4)
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        _activation_summary(bbox_pred)
        print ('bbox_pred.shape=',bbox_pred.get_shape())


# ignore the gesture prediction part and see how will the training go
        """ gesture prediction """
        #batch*H*W*3 shape=(batch,1,1,3)
        
        gesture_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
        gesture_pred = slim.fully_connected(gesture_pred, num_outputs=3,scope="gesture_fc",activation_fn=tf.nn.softmax)
        #thinking about change the activation fn to sigmoid or softmax?
        #Here trying to normalize: gesture_pred = gesture_pred / abs(gesture_pred) 

        _activation_summary(gesture_pred)
        print ('gesture_pred.shape=',gesture_pred.get_shape())
        

        #cls_prob_original = conv4_1 
        #bbox_pred_original = bbox_pred
        if training:
            #batch*2
            # calculate classification loss

            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob') # remove all size 1 dimensions, conv4_1 is the output tensor, [1,2] are axes
            cls_loss = cls_ohem(cls_prob,label)

            #batch*4
            # cal bounding box error, squared sum error
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            print("bbox_pred ", bbox_pred.get_shape())
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            #batch*3
            
            gesture_pred = tf.squeeze(gesture_pred,[1,2],name="gesture_pred")
            print("gesture_pred ", gesture_pred.get_shape())
            gesture_loss = gesture_ohem(gesture_pred,gesture_target,label)
            
            accuracy = cal_accuracy(cls_prob,label)
            # regularization loss: L2_loss
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,gesture_loss,L2_loss,accuracy
        #test

        else:
            cls_pro_test = tf.squeeze(conv4_1, name='cls_prob')
            print("cls_pro_test: ", cls_pro_test.get_shape())
            bbox_pred_test = tf.squeeze(bbox_pred, name='bbox_pred')
            print("bbox_pred_test: ", bbox_pred_test.get_shape())
            
            gesture_pred_test = tf.squeeze(gesture_pred,name="gesture_pred")
            print("gesture_pred_test: ", gesture_pred_test.get_shape())
            
            return cls_pro_test,bbox_pred_test,gesture_pred_test

        # #inference
        # else:
        #     #when inference,batch_size = 1

        #     cls_pro_test = tf.squeeze(conv4_1, axis=0)
        #     bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
        #     gesture_pred_test = tf.squeeze(gesture_pred,axis=0)
        #     return cls_pro_test,bbox_pred_test,gesture_pred_test


def R_Net(inputs,label=None,bbox_target=None,gesture_target=None,training=False):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print ("input shape: ", inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        print ("conv1 shape: ", net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print("pool1 shape: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        print("conv2 shape: ", net.get_shape())
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        print("pool2 shape: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        print("conv3 shape: ", net.get_shape())
        fc_flatten = slim.flatten(net)
        print("flatten: ", fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        print("fc1 shape: ", fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print("cls_prob shape: ", cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print("bbox_pred shape: ", bbox_pred.get_shape())
        #batch*3
        gesture_pred = slim.fully_connected(fc1,num_outputs=10,scope="gesture_pre_fc",activation_fn=None)
        gesture_pred = slim.fully_connected(gesture_pred, num_outputs=3,scope="gesture_fc",activation_fn=tf.nn.softmax)
        print("gesture_pred shape: ", gesture_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label,training)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            gesture_loss = gesture_ohem(gesture_pred,gesture_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,gesture_loss,L2_loss,accuracy

        else:
            return cls_prob,bbox_pred,gesture_pred
    
    """ -------NEED MODIFICATION BEFORE NEXT STAGE TRAINING-------- """
# PLEASE MODIFY THE FN BEFORE TRAINING ONET!!!
def O_Net(inputs,label=None,bbox_target=None,gesture_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        print("inputs: ", inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print("conv1: ", net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print("pool1: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print("conv2: ", net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print("pool2: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print("conv3: ", net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print("pool3: ", net.get_shape())
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print("conv4: ", net.get_shape())
        fc_flatten = slim.flatten(net)
        print("flatten: ", fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        print("fc1: ", fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print("cls prob: ", cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print("bbox pred: ", bbox_pred.get_shape())
        #batch*3
        gesture_pred = slim.fully_connected(fc1,num_outputs=10,scope="gesture_pre_fc",activation_fn=None)
        gesture_pred = slim.fully_connected(gesture_pred,num_outputs=3,scope="gesture_fc",activation_fn=tf.nn.softmax)
        print("gesture pred: ", gesture_pred.get_shape())
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label,training)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            gesture_loss = gesture_ohem(gesture_pred, gesture_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,gesture_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,gesture_pred
        
