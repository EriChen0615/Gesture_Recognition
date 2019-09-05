from mtcnn_model import P_Net
from train import test

def test_PNet(base_dir, prefix, display):
    net_factory = P_Net
    test(net_factory, prefix, base_dir, display=display)

if __name__ == '__main__':
    #data path
    
    base_dir = '../Dataset/Testing/imglists/PNet'
    display = 50
    model_name = 'MTCNN'
    #model_path = '../data/%s_model/PNet/PNet' % model_name
    #with gesture
    model_path = '../Model/{0}/PNet'.format(model_name)
            
    prefix = model_path

    print("--------------------Start Testing---------------------\n")

    
    test_PNet(base_dir, prefix, display)

    print("-------------------Testing Finished--------------------\n")