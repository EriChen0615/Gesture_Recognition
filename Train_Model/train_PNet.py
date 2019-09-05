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
    base_dir = '../Dataset/Training/imglists/PNet'
    model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with gesture
    model_path = '../Model/{0}/PNet'.format(model_name)
            
    prefix = model_path
    end_epoch = 300
    display = 100

    """change base learning rate here!"""
    lr = 0.1 #was 0.001

    print("------------------Training Started-------------------\n")
    train_PNet(base_dir, prefix, end_epoch, display, lr)

    print("------------------Training Finished-------------------\n")
    print("--------------------Start Testing---------------------")

    base_dir_ = '../Dataset/Testing/imglists/PNet'
    display = 200
    test_PNet(base_dir_, prefix, end_epoch, display)
 


