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
from util.util import compute_results,compute_results2, visualize,visualize_gt
from sklearn.metrics import confusion_matrix
from model.RT_CAN import GasSegNet
from tqdm import tqdm

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--weight_name', '-w', type=str, default='ResNet50')
parser.add_argument('--file_name', '-f', type=str, default='best.pth')
parser.add_argument('--dataset_split', '-d', type=str,
                    default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=512)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='F:\\WJ\\GasSeg\\dataset')
args = parser.parse_args()
#############################################################################################

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "gas"]
    testing_results_file = os.path.join(model_dir, 'testing_results.txt')
    with torch.no_grad():
        for it, (images, labels, names) in tqdm(enumerate(test_loader)):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            # BBS使用双输出
            logit, logits,_,_,_ = model(images)
            logit_mix = (logit + logits)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logit_mix.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[
                                    0, 1])
            conf_total += conf
    precision, recall, IoU, F1, F2, _ = compute_results2(conf_total)

    if epo == 0:
        with open(testing_results_file, 'w') as f:
            # f.write(
            #     "# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump, average(nan_to_num). (Acc %, IoU %)\n")
            f.write(
                "# epoch: unlabeled, gas, average(nan_to_num), (precision %,(recall)Acc %, IoU %, F1 ,F2)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f, %0.4f|' % (100*precision[i], 100 * recall[i], 100 * IoU[i], 100* F1[i], 100* F2[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f, %0.4f| \n' % (
            100 * np.mean(np.nan_to_num(precision)), 100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU)), 100 * np.mean(np.nan_to_num(F1)), 100 * np.mean(np.nan_to_num(F2))))
    print('saving testing results.')
    # with open(testing_results_file, "r") as file:
    #     writer.add_text('testing_results',
    #                     file.read().replace('\n', '  \n'), epo)
    # return IoU.mean(), recall.mean()


if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model_dir = os.path.join('./checkpoints/', args.weight_name)
    model_list = os.listdir(model_dir)
    num = 0

    batch_size = 1  # do not change this parameter!
    test_dataset = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height,
                              input_w=args.img_width)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    if args.weight_name == "ResNet50":
        num_resnet_layers = 50
    elif args.weight_name == "ResNet152":
        num_resnet_layers = 152
    else:
        sys.exit('no such type model.')

    # 遍历文件列表
    for index,model_name in tqdm(enumerate(model_list)):
        # 检查文件名是否以 '.pth' 结尾
        if model_name.endswith('.pth'):
            model_file = os.path.join(model_dir, model_name)
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

            testing(num, model, test_loader)
            num += 1
    