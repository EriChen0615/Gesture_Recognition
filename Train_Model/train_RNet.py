<<<<<<< HEAD
from mtcnn_model import P_Net
from train import train, test


def train_PNet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param base_dir: tfrecord path
    :param prefix:
    :param end_epoch: max epoch for training
    :param display:
    :param lr: learning rate
    :return:
    """
    net_factory = P_Net
    train(net_factory,prefix, end_epoch, base_dir, display=display, base_lr=lr)

def test_PNet(base_dir, prefix, end_epoch, display):
    net_factory = P_Net
    test(net_factory, prefix, end_epoch, base_dir, display=display)


if __name__ == '__main__':
    #data path
#    base_dir = '../Dataset/Training/imglists/PNet'
    model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with gesture
    model_path = '../Model/{0}/RNet'.format(model_name)
            
    prefix = model_path
    end_epoch = 30
    display = 100

    """change base learning rate here!"""
    lr = 0.1 #was 0.001

    print("------------------Training Started-------------------\n")
    train_RNet(base_dir, prefix, end_epoch, display, lr)

    print("------------------Training Finished-------------------\n")
    print("--------------------Start Testing---------------------")

    base_dir_ = '../Dataset/Testing/imglists/PNet'
    display = 200
    test_RNet(base_dir_, prefix, end_epoch, display)
 


=======
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
>>>>>>> d0079d24789f2f6c44233af480cdf8a2a60f9071
