import os
import argparse
import time
import datetime
import sys
import shutil
import stat
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from util.util import compute_results, show
from sklearn.metrics import confusion_matrix
from model.RT_CAN import GasSegNet
import PIL

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--weight_name', '-w', type=str, default='ResNet50')
# parser.add_argument('--weight_name', '-w', type=str, default='2023.6.23.20.51/')
parser.add_argument('--file_name', '-f', type=str, default='best.pth')
parser.add_argument('--dataset_split', '-d', type=str,
                    default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=512)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset')
args = parser.parse_args()
#############################################################################################

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    model_dir = os.path.join('./checkpoints/', args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')

    conf_total = np.zeros((args.n_class, args.n_class))
    if args.weight_name == "ResNet50":
        num_resnet_layers = 50
    elif args.weight_name == "ResNet152":
        num_resnet_layers = 152
    else:
        sys.exit('no such type model.')
    model = GasSegNet(args.n_class,num_resnet_layers)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(
        model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()

    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    for name, param in pretrained_weight.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param)
    print('done!')

    ave_time_cost = 0.0

    model.eval()
    file_path = "dataset\\images\\00665.png"
    image1 = np.asarray(PIL.Image.open(file_path))
    image = np.asarray(PIL.Image.fromarray(image1).resize((640, 512)), dtype=np.float32).transpose((2,0,1))/255
    image = torch.tensor(image).unsqueeze(0)
    images = Variable(image).cuda(args.gpu)
    logit, logits, _, _, _ = model(images)
    logit_mix = logit + logits
    show(image_name=file_path, predictions=logit_mix.argmax(1), weight_name='Pred_' + args.weight_name,origin = image1)