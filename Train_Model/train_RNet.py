from mtcnn_model import P_Net
from train import train, test


def parse_args():
    parser = argparse.ArgumentParser(description='Train RNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name',dest='model_name',help='the name of the model',default='MTCNN-test')
    parser.add_argument('--tfrecord_dir',dest='tfrecord_dir',help='directory for tfrecord',default='../Dataset/Training/imglists/RNet')
    parser.add_argument('--base_lr',dest='lr',help='starting learn rate for training',default=0.1)
    parser.add_argument('--end_epoch',dest='end_epoch',help='end epoch',default=30)
    args = parser.parse_args()
    return args


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
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr, with_gesture)

if __name__ == '__main__':

    args = parse_args()
    #data path
    base_dir = args.tfrecord_dir
    model_name = args.model_name
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with gesture
    model_path = '../Model/{0}/RNet'.format(model_name)
    # model_path = '../Model/{0}/PNet/PNet'.format(model_name)
    
    prefix = model_path
    end_epoch = int(args.end_epoch)
    display = 20

    with_gesture = False
    lr = float(args.lr)
    train_RNet(base_dir, prefix, end_epoch, display, lr, with_gesture)
