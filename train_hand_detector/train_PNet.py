from mtcnn_model import P_Net
from train import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name',dest='model_name',help='the name of the model',default='MTCNN')
    parser.add_argument('--tfrecord_dir',dest='tfrecord_dir',help='directory for tfrecord',default='../Dataset/Training/imglists/PNet')
    parser.add_argument('--base_lr',dest='lr',help='starting learn rate for training',default=0.1)
    parser.add_argument('--end_epoch',dest='end_epoch',help='end epoch',default=30)
    args = parser.parse_args()
    return args

def train_PNet(base_dir, prefix, end_epoch, display, lr, with_gesture):
    """
    train PNet
    :param base_dir: tfrecord path
    :param prefix:
    :param end_epoch: max epoch for training
    :param display:
    :param lr: learning rate
    :return:
    """
    net_factory = P_Net # P_Net is a function defined in mtcnn_model
    train(net_factory,prefix, end_epoch, base_dir, display=display, base_lr=lr, with_gesture)



if __name__ == '__main__':
    args = parse_args()
    #data path
    base_dir = args.tfrecord_dir
    model_name = args.model_name
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with gesture
    model_path = '../Model/{0}/PNet'.format(model_name)
    # model_path = '../Model/{0}/PNet/PNet'.format(model_name)
    
    prefix = model_path
    end_epoch = int(args.end_epoch)
    display = 20

    """change base learning rate here!"""
    lr = float(args.lr) #was 0.001

    print("------------------Training Started-------------------\n")
    train_PNet(base_dir, prefix, end_epoch, display, lr, with_gesture)

    print("------------------Training Finished-------------------\n")

 


