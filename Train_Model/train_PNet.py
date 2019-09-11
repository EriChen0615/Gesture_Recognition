from mtcnn_model import P_Net
from train import train


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
    #data path
    base_dir = '../Dataset/Training/imglists/PNet'
    model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with gesture
    model_path = '../Model/{0}/PNet_NO_Landmark/PNet'.format(model_name)
    # model_path = '../Model/{0}/PNet/PNet'.format(model_name)
    
    prefix = model_path
    end_epoch = 30
    display = 20

    """change base learning rate here!"""
    lr = 0.1 #was 0.001
    with_gesture = False

    print("------------------Training Started-------------------\n")
    train_PNet(base_dir, prefix, end_epoch, display, lr, with_gesture)

    print("------------------Training Finished-------------------\n")

 


