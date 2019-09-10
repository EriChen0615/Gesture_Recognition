from mtcnn_model import O_Net
from train import train


def train_ONet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = O_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    base_dir = '../Dataset/Training/no_LM48/imglists/ONet'
    model_name = 'MTCNN'
    model_path = '../Model/%s/ONet_No_Landmark/ONet' % model_name
    prefix = model_path
    end_epoch = 116 #was 22
    display = 100
    lr = 0.01 #was 0.001
    train_ONet(base_dir, prefix, end_epoch, display, lr)
