#coding:utf-8
from mtcnn_model import R_Net
from train import train


def train_RNet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = R_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    base_dir = '../Dataset/Training/no_LM24/imglists/RNet'

    model_name = 'MTCNN'
    model_path = '../Model/%s/RNet_No_Landmark/RNet' % model_name
    prefix = model_path
    end_epoch = 22
    display = 100
    lr = 0.001
    train_RNet(base_dir, prefix, end_epoch, display, lr)
